'''
 * Copyright (c) 2025, VRR-ScenarioInfer.
 * All rights reserved.
 * By Yubing Li
'''
import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from models.VRR_test import scenarioinfer
import utils
from utils import cosine_lr_schedule
from data import create_dataset, create_sampler, create_loader
from data.utils import save_result_peppa

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score 
from rouge_score import rouge_scorer
from pycocoevalcap.cider.cider import Cider
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import zoom


def train(model, data_loader, optimizer, epoch, device):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Caption Epoch: [{}]'.format(epoch)
    print_freq = 50

    for i, (image, caption, annotition) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device)   
        
        loss = model(image, caption, annotition)        

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    

        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     

    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}  

@torch.no_grad()
def evaluate(model, data_loader, device, config):
    model.eval() 
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Caption generation:'
    print_freq = 10

    result = []
    references = []
    all_image_paths = []

    for (images, captions, current_image_paths, annotations) in metric_logger.log_every(data_loader, print_freq, header):
        images = images.to(device)

        for i in range(images.size(0)):
            use_image = images[i].unsqueeze(0)
            use_annotation = annotations[4][i]
            generated_caption = model.generate(use_image, use_annotation, num_beams=config['num_beams'], 
                                            max_length=config['max_length'], min_length=config['min_length'])
            result.extend(generated_caption)

            # Add the current reference and image path to their respective lists
            references.append(use_annotation)
            all_image_paths.append(current_image_paths[0][i])

    # Calculate BLEU@4 score
    bleu_scores = []
    smoothing_function = SmoothingFunction().method4
    
    for ref, gen in zip(references, result):
        if isinstance(ref, list):
            ref = ' '.join(ref)
        ref_tokens = nltk.word_tokenize(ref)
        gen_tokens = nltk.word_tokenize(gen)
        bleu_scores.append(sentence_bleu([ref_tokens], gen_tokens, smoothing_function=smoothing_function))
    
    avg_bleu_score = sum(bleu_scores) / len(bleu_scores)
    
    # Calculate METEOR score      
    meteor_scores = []
    for ref, gen in zip(references, result):
        if isinstance(ref, list):
            ref = ' '.join(ref)
        ref_tokens = nltk.word_tokenize(ref)
        gen_tokens = nltk.word_tokenize(gen)
        meteor_scores.append(single_meteor_score(ref_tokens, gen_tokens))
    
    avg_meteor_score = sum(meteor_scores) / len(meteor_scores)
    
    # Calculate ROUGE score
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = []
    
    for ref, gen in zip(references, result):
        if isinstance(ref, list):
            ref = ' '.join(ref)
        scores = rouge_scorer_obj.score(ref, gen)
        rouge_scores.append((scores['rouge1'].fmeasure + scores['rouge2'].fmeasure + scores['rougeL'].fmeasure) / 3)
    
    avg_rouge_score = sum(rouge_scores) / len(rouge_scores)
    
    # Calculate CIDEr score
    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score({i: [ref if isinstance(ref, str) else ' '.join(ref)] for i, ref in enumerate(references)}, 
                                                {i: [gen if isinstance(gen, str) else gen[0]] for i, gen in enumerate(result)})

    # Load pre-trained models for sentence similarity and NLI
    model = AutoModelForSequenceClassification.from_pretrained("./deberta-base-mnli").to(device)
    model.eval()

    # Calculate BERTScore
    tokenizer_bert = AutoTokenizer.from_pretrained("./bert-base-uncased", local_files_only=True)
    bert_model = AutoModel.from_pretrained("./bert-base-uncased", local_files_only=True).to(device)
    bert_model.eval()

    def compute_bertscore_single(ref, hyp, model, tokenizer, device='cuda'):
        with torch.no_grad():
            inputs_ref = tokenizer(ref, return_tensors="pt", truncation=True, max_length=128).to(device)
            inputs_hyp = tokenizer(hyp, return_tensors="pt", truncation=True, max_length=128).to(device)

            outputs_ref = model(**inputs_ref)[0]
            outputs_hyp = model(**inputs_hyp)[0]

            ref_emb = outputs_ref[0, 1:-1]  
            hyp_emb = outputs_hyp[0, 1:-1]

            ref_emb = F.normalize(ref_emb, p=2, dim=1)
            hyp_emb = F.normalize(hyp_emb, p=2, dim=1)

            sim_matrix = torch.mm(ref_emb, hyp_emb.transpose(0, 1))

            recall = sim_matrix.max(dim=1)[0].mean()
            precision = sim_matrix.max(dim=0)[0].mean()

            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = torch.tensor(0.0).to(device)

            return f1.item()

    bertscore_scores = []
    for ref, hyp in zip(references, result):
        score = compute_bertscore_single(ref, hyp, bert_model, tokenizer_bert, device=device)
        bertscore_scores.append(score)

    avg_bertscore = sum(bertscore_scores) / len(bertscore_scores)

    return result, references, all_image_paths, avg_bleu_score, avg_meteor_score, avg_rouge_score, cider_score, avg_bertscore


def main(args, config):
    device = torch.device(args.device)
        
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating captioning dataset")
    train_dataset, val_dataset, test_dataset = create_dataset('caption_peppa', config)  

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler([train_dataset,val_dataset,test_dataset], [False,False,False], num_tasks, global_rank)         
    else:
        samplers = [None, None, None]
    
    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset],samplers,
                                                          batch_size=[config['batch_size']]*3,num_workers=[4,0,0],
                                                          is_trains=[True, False, False], collate_fns=[None,None,None])         

    #### Model #### 
    print("Creating model")
    model = scenarioinfer(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'], 
                           vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], 
                           prompt=config['prompt'])
    
    model = model.to(device)   
    

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {count_parameters(model):,}")

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])
            
    best_epoch = 0

    print("Start training")
    start_time = time.time()    
    for epoch in range(0, config['max_epoch']):
        if not args.evaluate:        
            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])
            train_stats = train(model, train_loader, optimizer, epoch, device) 
        
        test_result, references, image_paths, avg_bleu_score, avg_meteor_score, avg_rouge_score, cider_score, Bert_score= evaluate(model, test_loader, device, config)  
        test_result_file, test_metric_file = save_result_peppa(test_result, references, image_paths, avg_bleu_score, avg_meteor_score, avg_rouge_score, cider_score, Bert_score, args.result_dir, 'test_epoch%d' % epoch)     

        print('Evaluating Test caption...')

        if args.evaluate:            
            log_stats = {}

            if test_result_file:
                log_stats.update({f'test_{k}': v for k, v in test_result_file.items()})
            else:
                print("Warning: test_result_file is empty or None.")
                
            # Check if log_stats can be serialized
            try:
                log_stats_str = json.dumps(log_stats)
            except TypeError as e:
                print(f"Error serializing log_stats: {e}")
                log_stats_str = "{}"  # Default value

            with open(os.path.join(args.output_dir, "evaluate.txt"), "a") as f:
                f.write(log_stats_str + "\n") 

            log_stats_m = {**{f'test_{k}': v for k, v in test_metric_file.items()}}
            with open(os.path.join(args.output_dir, "evaluate_metric.txt"), "a") as f:
                f.write(json.dumps(log_stats_m) + "\n") 

        else:             
            save_obj = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config,
                'epoch': epoch,
            }
                    
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,
                        'best_epoch': best_epoch}
            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")     

        if args.evaluate: 
            break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 

def save_checkpoint(model, optimizer, epoch, save_dir, filename='checkpoint.pth.tar'):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    save_path = os.path.join(save_dir, filename)
    torch.save(state, save_path)
    print(f"Model checkpoint saved at {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='')
    parser.add_argument('--output_dir', default='')        
    parser.add_argument('--evaluate', action='store_true')    
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=False, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)
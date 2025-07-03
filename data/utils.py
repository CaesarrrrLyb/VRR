import re
import json
import os

import torch
import torch.distributed as dist

import utils

def pre_caption(caption, max_words=50):

    caption = caption.lower()
    

    caption = re.sub(r"\s{2,}", ' ', caption)
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')


    caption_words = caption.split(' ')
    if len(caption_words) > max_words:
        caption = ' '.join(caption_words[:max_words])
            
    return caption

def pre_question(question,max_ques_words=50):
    question = re.sub(
        r"([.!\"()*#:;~])",
        '',
        question.lower(),
    ) 
    question = question.rstrip(' ')
    
    #truncate question
    question_words = question.split(' ')
    if len(question_words)>max_ques_words:
        question = ' '.join(question_words[:max_ques_words])
            
    return question

def save_result_peppa(result, reference, image_paths, avg_bleu_score, avg_meteor_score, avg_rouge_score, cider_score, Bert_score, result_dir, filename):

    combined_results = [{'result': res, 'reference': ref, 'image_paths': img} for res, ref, img in zip(result, reference, image_paths)]


    results_file = os.path.join(result_dir, f'{filename}_results.json')
    with open(results_file, 'w') as f:
        json.dump(combined_results, f)


    metrics_file = os.path.join(result_dir, f'{filename}_metrics.json')
    metrics = {
        'Bleu_4': avg_bleu_score,
        'avg_meteor_score': avg_meteor_score,
        'avg_rouge_score': avg_rouge_score,
        'CIDEr': cider_score,
        'Bert_score': Bert_score,
        }

    with open(metrics_file, 'w') as f:
        json.dump(metrics, f)

    print(f'Results and references saved to {results_file}')
    print(f'Metrics saved to {metrics_file}')

    return results_file, metrics_file


def save_result(result, result_dir, filename, remove_duplicate=''):
    result_file = os.path.join(result_dir, '%s_rank%d.json'%(filename,utils.get_rank()))
    final_result_file = os.path.join(result_dir, '%s.json'%filename)
    
    json.dump(result,open(result_file,'w'))

    # dist.barrier()

    if utils.is_main_process():   
        # combine results from all processes
        result = []

        for rank in range(utils.get_world_size()):
            result_file = os.path.join(result_dir, '%s_rank%d.json'%(filename,rank))
            res = json.load(open(result_file,'r'))
            result += res

        if remove_duplicate:
            result_new = []
            id_list = []    
            for res in result:
                if res[remove_duplicate] not in id_list:
                    id_list.append(res[remove_duplicate])
                    result_new.append(res)
            result = result_new             
                
        json.dump(result,open(final_result_file,'w'))            
        print('result file saved to %s'%final_result_file)

    return final_result_file





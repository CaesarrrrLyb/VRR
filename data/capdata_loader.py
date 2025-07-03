import os
import json
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
from PIL import Image
from data.utils import pre_caption
import torch
    
class capdata_train(Dataset):
    def __init__(self, transform, image_root, ann_root, max_words=30, prompt=''):
        # filename = 'grouped_peppa_lcc_try.json'
        # filename = 'grouped_peppa_lcc_train_ann.json'
        filename = 'coinic_train.json'
        self.annotation = json.load(open(os.path.join(ann_root, filename), 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.prompt = prompt

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann_group = self.annotation[index]

        images = []
        captions = []
        annotations = []

        for ann in ann_group:
            image_path = os.path.join(self.image_root, ann['image'])
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
            caption = self.prompt + pre_caption(ann['caption'], self.max_words)

            images.append(image)
            captions.append(caption)
            annotations.append(ann.get('Annotation', None))


        images = torch.stack(images, dim=0)

        return images, captions, annotations


class capdata_eval(Dataset):
    def __init__(self, transform, image_root, ann_root, split, max_words=30):
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): Directory to store the annotation file
        split (string): val or test
        '''

        # filenames = {'val': 'grouped_peppa_lcc_test_ann.json', 'test': 'grouped_peppa_lcc_test_ann.json'}
        filenames = {'val': 'coinic_test.json', 'test': 'coinic_test.json'}
        # filenames = {'val': 'grouped_peppa_lcc_try.json', 'test': 'grouped_peppa_lcc_try.json'}
        
        self.annotation = json.load(open(os.path.join(ann_root, filenames[split]), 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann_group = self.annotation[index]

        images = []
        captions = []
        image_paths = []  
        annotations = []

        for ann in ann_group:
            image_path = os.path.join(self.image_root, ann['image'])
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
            caption = pre_caption(ann['caption'], self.max_words)

            images.append(image)
            captions.append(caption)
            image_paths.append(ann['image_id'])
            annotations.append(ann.get('Annotation', None))

        images = torch.stack(images, dim=0)

        return images, captions, image_paths, annotations



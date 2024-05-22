from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
from PIL import Image
import numpy as np
import math
import shutil
import json
import time

from src.data.components.aug.wrapper_v2 import Augmenter
from src.data.components.aug.vietocr_aug import ImgAugTransform

def delete_contents_of_folder(folder_path):
    try:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        print(f"Deleted all contents of the folder: {folder_path}")
    except Exception as e:
        print(f"Error deleting contents of the folder: {folder_path}")

class XLADataset(Dataset):
    ''' This dataset only loads images from files into numpy arrays '''

    def __init__(
        self, 
        data_dir: str, 
        manifest: str,
        task: str,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.task = task
        
        self.get_vocab(manifest)
        self.samples = self.load_data(manifest)
        
    
    def get_vocab(self, manifest):
        with open(manifest, "r") as file:
            samples = json.load(file)["train"]
        
        keys = list(samples.keys())
        self.vocab = dict(zip(keys, range(len(keys))))
        print(f"Length of vocab: {len(self.vocab)}")
        print(self.vocab)

    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, index) -> tuple:
        
        filename, label = self.samples[index]
        # print("data: ", filename, label)
        # open & process image
        image_path = os.path.join(self.data_dir, self.task)
        image_path = os.path.join(image_path, filename)
        
        image = np.array(Image.open(image_path).convert("RGB"))

        return {'filename': filename, 'image': image, 'label': label}

    def load_data(self, manifest):
        with open(manifest, "r") as file:
            self.id2sample = json.load(file)[self.task]
        
        keys = list(self.id2sample.keys())
        
        key_num = [len(self.id2sample[id]) for id in list(self.id2sample.keys())]
        ranges = [0] + np.cumsum(key_num).tolist()
        key_ranges = [[ranges[i-1], ranges[i]] for i in range(1, len(ranges))]
        # get id distribution of space for upsampling 
        self.ranges = dict(zip(keys, key_ranges))

        # prepares token 
        sample2id = []
        
        for key in keys: 
            filenames = self.id2sample[key]
            for fname in filenames:
                sample2id.append([fname, self.vocab[key]])   
        # # filenames = sum(list(self.id2sample.values()), [])
        # # sample_keys = sum([[id] * len(self.id2sample[id]) for id in self.id2sample.keys()], [])
        
        # sample2id = list(zip(filenames, sample_keys))

        return sample2id
    
    def __len__(self):
        return len(self.samples)
    
    def num_classes(self):
        return len(self.vocab)

class XLATransformedDataset(Dataset):
    def __init__(
            self,
            dataset: XLADataset, 
            augmenter: Augmenter,
            transform: ImgAugTransform,
            p = [0.6, 0.2, 0.2],
    ):
        assert augmenter is not None
        self.dataset = dataset
        # shape transformation
        self.augmenter = augmenter
        self.augmenter.task = self.dataset.task

        # pixel transform
        self.transform = transform
        self.p = p

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        sample = self.dataset[index]
        filename = sample['filename']
        image = sample['image']
        label = sample['label']
        
        # first of all, transform the image before applying
        if self.p != [0.0, 0.0, 0.0]:
            image = self.augmenter.full_augment(image, 
                                                choice=self.p,
                                                fname=filename,
                                                borderMode='native')
    
        if isinstance(self.transform, ImgAugTransform):
            image = self.transform(image)

        return {'filename': filename, 'image': image, 'label': label}

    def num_classes(self):
        return self.dataset.num_classes()
 
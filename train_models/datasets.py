import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image
from torchvision.models import efficientnet_b4
import glob
import sys
import os
sys.path.append("../")
from utils import file_to_dict, get_imagenet100_synsets

def syn_dataset_helper(data_dir, fname, map_dict):
    synsets = get_imagenet100_synsets(fname)

    items = []
    for index, synset in enumerate(synsets):
        syn_id = "n{:08d}".format(synset)
        syn_info = map_dict[syn_id]
        class_num = int(syn_info[0])
        class_name = syn_info[1].strip().lower()
        
        syn_id = "{:08d}".format(synset)
        base_dir = data_dir+class_name+"/"
        files = os.listdir(base_dir)
        idx = np.arange(30)
        np.random.shuffle(idx)
        for i in idx:
            items.append((base_dir+files[i], index))
    return items

def real_dataset_helper(data_dir, fname, map_dict):
    synsets = get_imagenet100_synsets(fname)

    items = []
    for index, synset in enumerate(synsets):
        syn_id = "n{:08d}".format(synset)
        class_num = int(map_dict[syn_id][0])
        base_dir = data_dir+syn_id+"/"
        files = os.listdir(base_dir)
        idx = np.arange(40)
        np.random.shuffle(idx)
        for i in idx:
            items.append((base_dir+files[i], index))
    return items

class SyntheticImageNetDataset(Dataset):
    def __init__(self,data_dir,map_fname,syn_fname):
        self.data_dir = data_dir
        self.map_dict = file_to_dict(map_fname)
        self.items = syn_dataset_helper(data_dir,syn_fname,self.map_dict)
        self.transform = transforms.Compose([
        	transforms.Resize(256),
        	transforms.RandomHorizontalFlip(0.5),
        	transforms.RandomCrop(224),
        	transforms.Lambda(self.normalize)])
    
    def normalize(self, x):
    	x /- 127.5
    	x -= 1
    	return x 	

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fpath, cls = self.items[idx]

        
        img = read_image(fpath).float()
        
        if img.shape[0] == 1:
            img = img.repeat(3,1,1)
        
        img = self.transform(img)

        return img, cls

class RealImageNetDataset(Dataset):
    def __init__(self,data_dir,map_fname, syn_fname):
        self.data_dir = data_dir
        self.map_dict = file_to_dict(map_fname)
        self.items = real_dataset_helper(data_dir,syn_fname,self.map_dict)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomCrop(224),
            transforms.Lambda(self.normalize)])
        	
    def normalize(self, x):
        x /= 127.5
        x -= 1
        return x

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fpath, cls = self.items[idx]

        img = read_image(fpath).float()
        
        if img.shape[0] == 1:
            img = img.repeat(3,1,1)
        
        img = self.transform(img)

        #label = np.zeros(100)
        #label[cls] = 1.0

        return img, cls


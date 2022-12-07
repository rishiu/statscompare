import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torchvision.io import read_image
from torchvision.models import efficientnet_b4
import glob
import sys
import os
sys.path.append("../")
from utils import file_to_dict, get_common_synsets

class SyntheticImageNetDataset(Dataset):
    def __init__(self,data_dir,fname):
        self.data_dir = data_dir
        self.map_dict = file_to_dict(fname)

    def __len__(self):
        return len(glob.glob("*.jpg"))

    def __getitem__(self, idx):
        img_files = glob.glob("*.jpg")
        file = img_files[idx]

        name = file[:-4]
        class_num = int(self.map_dict[name][0])

        label = np.zeros(1000)
        label[class_num] = 1.0

        img = read_image(self.data_dir+file)
        img = img[:3,:,:]

        return img, label

def dataset_helper(data_dir, fname, map_dict):
    synsets = get_common_synsets(fname)

    items = []
    for synset in synsets:
        class_num = map_dict[synset]
        files = os.listdir(data_dir+"n0"+str(synset)+"/")
        idx = np.random.randint(0,len(files),size=10)
        for i in idx:
            items.append((files[i], class_num))
    return items

class RealImageNetDataset(Dataset):
    def __init__(self,data_dir,fname):
        self.data_dir = data_dir
        self.map_dict = file_to_dict(fname)
        self.items = dataset_helper(data_dir,fname,self.map_dict)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fpath, cls = self.items[idx]

        img = read_image(fpath)[:3,:,:]

        label = np.zeros(1000)
        label[cls] = 1.0

        return img, label


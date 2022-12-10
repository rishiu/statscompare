import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b4
import torch.optim as optim
import os
import sys
from tqdm import tqdm
import pytorch_lightning as pl
from datasets import SyntheticImageNetDataset, RealImageNetDataset
sys.path.append("../")
from utils import file_to_file2

class ImageNetEfficientNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = efficientnet_b4()

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self.model(x)
	
        loss = F.cross_entropy(z,y)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.0005)
        return optimizer

    def validation_step(self, batch, batch_idx):
        x, y = batch

        z = self.model(x)

        loss = F.cross_entropy(y,z)
        self.log('val_loss', loss)

        y_c = y.argmax(1)
        z_c = z.argmax(1)

        accuracy = (y_c == z_c).count_nonzero()
        self.log('accuracy', accuracy)

def train(epochs, train_dir, val_dir, flag="S", save_dir="./"):
    dataset = SyntheticImageNetDataset if flag == "S" else RealImageNetDataset

    model = ImageNetEfficientNet()
    
    train_dataset = dataset(train_dir, "../clsloc_dict.txt", "../imagenet20_classes.txt")
    #val_dataset = dataset(val_dir)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=1, num_workers=4)
    #val_dataloader = DataLoader(val_dataset)

    trainer = pl.Trainer(accelerator='gpu', devices=[0,1], max_epochs=epochs, default_root_dir=save_dir)
    trainer.fit(model, train_dataloader)
    
if __name__ == "__main__":
    #file_to_file2("../map_clsloc.txt", "../nameloc_dict.txt")
    train(100, sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])


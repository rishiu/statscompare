import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b4
import torch.optim as optim
import os
from tqdm import tqdm
import pytorch_lightning as pl
from datasets import SyntheticImageNetDataset, RealImageNetDataset

class ImageNetEfficientNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = efficientnet_b4()

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self.model(x)

        loss = F.cross_entropy(y,z)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
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
    
    train_dataset = dataset(train_dir)
    val_dataset = dataset(val_dir)

    train_dataloader = DataLoader(train_dataset)
    val_dataloader = DataLoader(val_dataset)

    trainer = pl.Trainer(accelerator='gpu', devices=[0], max_epochs=epochs, default_root_dir=save_dir)
    trainer.fit(model, train_dataloader, val_dataloader)


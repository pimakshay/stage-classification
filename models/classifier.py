import os
import numpy as np
import pandas as pd

import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from functools import partial

from utilities.dataloader import check_and_update_csv, split_train_val_test, get_dataset
from utilities.initialize_configs import *
from torch.utils.tensorboard import SummaryWriter
LOGS_PATH = os.path.join(os.getcwd(),'lightning_logs')
writer = SummaryWriter(LOGS_PATH)

class Classifier(pl.LightningModule):

    def __init__(self,
                 image_size=128,
                 channels=3,
                 num_classes=7,
                 loss_type="mse",
                 learning_rate=0.001,
                 max_tsteps=100000,
                 batch_size=16,
                 data_dir="",
                 nn_model=None
                 ) -> None:

        super().__init__()
        self.loss_type = loss_type
        self.image_size = image_size
        self.batch_size = batch_size
        self.channels = channels
        self.loss_type = loss_type
        self.learning_rate = learning_rate
        self.max_tsteps = max_tsteps
        img_dir = os.path.join(data_dir, 'foundation_images/foundation_images')

        self.dataframe = pd.read_csv(os.path.join(data_dir, 'stage_labels_corr.csv'),sep=";")
        # Extract camera index, timestamp, and structure index from the filename
        self.dataframe['camera'] = self.dataframe['imagename'].apply(lambda x: x.split('_')[0])
        self.dataframe['timestamp'] = self.dataframe['imagename'].apply(lambda x: x.split('_')[1][1:])
        self.dataframe['structure_index'] = self.dataframe['imagename'].apply(lambda x: int(x.split('_')[2].split('.')[0]))
        self.dataframe['numbered_label'] = self.dataframe['label'].apply(lambda x: int(x.split('-')[1][0]))

        class_labels = list(self.dataframe["label"].unique())
        self.class_labels = class_labels[-1:]+class_labels[:-1]
        self.class_labels_dict = {}
        for i, item in enumerate(class_labels):
            self.class_labels_dict[i] = item


        self.train_df, self.val_df, self.test_df = split_train_val_test(data_dir, self.dataframe, save_as_csv=False)                     
        self.train_dataset, self.val_dataset, self.test_dataset = get_dataset(self.train_df, self.val_df, self.test_df, img_dir=img_dir, img_size=image_size)

        self.model = instantiate_from_configs(nn_model)

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'cross_entropy':
            loss = torch.nn.functional.cross_entropy(pred,target)
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{self.loss_type}'")

        return loss

    def apply_model(self, x, y):
        model_out = self.model(x)

        loss_dict = {}
        loss = self.get_loss(model_out, y, mean=False)

        log_prefix = 'train' if self.training else 'val'

        loss_dict.update({f'{log_prefix}/loss': loss.mean()})
        
        loss_dict.update({f'trade-off/{log_prefix}': loss.mean()})

        return loss, loss_dict

    def forward(self, x, y, *args, **kwargs):
        return self.apply_model(x, y, *args, **kwargs)    

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    # def get_input(self, batch, k):
    #     x = batch
    #     x = x.to(memory_format=torch.contiguous_format).float()
    #     return x

    # def shared_step(self, batch):
    #     x = self.get_input(batch, self.first_stage_key)
    #     loss, loss_dict = self(x)
    #     return loss, loss_dict

    def training_step(self, batch, batch_idx):
        (x,y) = batch # x:image, y:label   
        loss, loss_dict = self(x,y)

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=False, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=False, on_epoch=True)
        
        lr = self.optimizers().param_groups[0]['lr']
        self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=False, on_epoch=True)
            
        # # N = self.epochs // 5 # linear lr-scheduling
        # if self.trainer.global_step < self.warmup_steps:
        #     lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.warmup_steps)
        #     for pg in self.optimizers().param_groups:
        #         pg["lr"] = lr_scale * self.learning_rate

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        (x,y) = batch # x:image, y:label   
        loss, loss_dict = self(x,y)     
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        
        return loss

    def train_dataloader(self):

        train_loader = torch.utils.data.DataLoader(
            dataset=self.train_dataset, batch_size=self.batch_size, num_workers=4, shuffle=False
        )

        return train_loader
    
    def val_dataloader(self):

        val_loader = torch.utils.data.DataLoader(
            dataset=self.val_dataset, batch_size=self.batch_size, num_workers=4, shuffle=False
        )
        return val_loader
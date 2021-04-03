
import time
from tqdm import tqdm

import numpy as np
import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset

import timm

from src.augmentation import *
from src.utils import *

class FlowerDataset(Dataset):
    def __init__(self, df,
                 shape,
                 transforms=None,
                 output_label=True,
                 one_hot_label=False,
                 image_name_col = "image_path",
                 label_col = "label"
                ):

        super().__init__()
        self.shape = shape
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms
        self.output_label = output_label
        self.one_hot_label = one_hot_label
        self.image_name_col = image_name_col
        self.label_col = label_col

        if output_label == True:
            self.labels = self.df[self.label_col].values
            if one_hot_label is True:
                self.labels = np.eye(self.df[self.label_col].max()+1)[self.labels]

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index: int):

        if self.output_label:
            target = self.labels[index]

        img  = get_img(self.df.loc[index][self.image_name_col])

        if self.transforms:
            img = self.transforms(image=img)['image']

        if self.output_label == True:
            return img, target
        else:
            return img

class FlowerImgClassifier(nn.Module):
    def __init__(self, model_arch, n_class, model_shape, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained)

        if model_shape == "classifier":
            n_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(n_features, n_class)
        elif model_shape == "head":
            n_features = self.model.head.in_features
            self.model.head = nn.Linear(n_features, n_class)
        elif model_shape == "fc":
            n_features = self.model.fc.in_features
            self.model.fc = nn.Linear(n_features, n_class)

    def forward(self, x):
        x = self.model(x)
        return x

class EarlyStopping:
    def __init__(self, patience):
        self.max_val_monitor = 1000
        self.min_val_monitor = -1000
        self.val_epoch = -1
        self.stop_count = 0
        self.patience = patience
        self.min_delta = 0

    # mode = "min" or "max"(val_loss, val_accuracy)
    def update(self, monitor, epoch, mode):
        if mode == "max":
            if monitor > self.min_val_monitor:
                self.min_val_monitor = monitor
                self.val_epoch = epoch
                self.stop_count = 0
            else:
                self.stop_count+=1
        else:
            if monitor < self.max_val_monitor:
                self.max_val_monitor = monitor
                self.val_epoch = epoch
                self.stop_count = 0
            else:
                self.stop_count+=1

        if self.stop_count > self.patience:
            return -1
        else:
            return 0

def prepare_dataloader(df, input_shape, trn_idx, val_idx, train_bs, valid_bs, num_workers):
    train_ = df.loc[trn_idx,:].reset_index(drop=True)
    valid_ = df.loc[val_idx,:].reset_index(drop=True)

    train_ds = FlowerDataset(train_, input_shape, transforms=get_train_transforms(input_shape), output_label=True, one_hot_label=False)
    valid_ds = FlowerDataset(valid_, input_shape, transforms=get_valid_transforms(input_shape), output_label=True)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=train_bs,
        pin_memory=True, # faster and use memory
        drop_last=False,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        valid_ds,
        batch_size=valid_bs,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
    )
    return train_loader, val_loader

def train_one_epoch(epoch, model, loss_fn, optimizer, train_loader, device, verbose_step, scheduler=None, schd_batch_update=False):
    model.train()
    scaler = GradScaler()

    t = time.time()
    running_loss = None

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for step, (imgs, image_labels) in pbar:
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).long()

        with autocast():
            image_preds = model(imgs)
            loss = loss_fn(image_preds, image_labels)

            scaler.scale(loss).backward()

            if running_loss is None:
                running_loss = loss.item()
            else:
                running_loss = running_loss * .99 + loss.item() * .01

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if scheduler is not None and schd_batch_update:
                scheduler.step()

            if ((step + 1) % verbose_step == 0) or ((step + 1) == len(train_loader)):
                description = f'epoch {epoch} loss: {running_loss:.4f}'
                pbar.set_description(description)

    print("train: "+ description)
    if scheduler is not None and not schd_batch_update:
        scheduler.step()

def valid_one_epoch(epoch, model, loss_fn, val_loader, device, verbose_step, scheduler=None, schd_loss_update=False):
    model.eval()

    t = time.time()
    loss_sum = 0
    sample_num = 0
    image_preds_all = []
    image_targets_all = []

    pbar = tqdm(enumerate(val_loader), total=len(val_loader))
    for step, (imgs, image_labels) in pbar:
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).long()

        image_preds = model(imgs)
        image_preds_all += [torch.argmax(image_preds, 1).detach().cpu().numpy()]
        image_targets_all += [image_labels.detach().cpu().numpy()]

        loss = loss_fn(image_preds, image_labels)

        loss_sum += loss.item()*image_labels.shape[0]
        sample_num += image_labels.shape[0]

        if ((step + 1) % verbose_step== 0) or ((step + 1) == len(val_loader)):
            description = f'epoch {epoch} loss: {loss_sum/sample_num:.4f}'
            pbar.set_description(description)

    print("valid "+ description)
    image_preds_all = np.concatenate(image_preds_all)
    image_targets_all = np.concatenate(image_targets_all)
    print('validation multi-class accuracy = {:.4f}'.format((image_preds_all==image_targets_all).mean()))

    if scheduler is not None:
        if schd_loss_update:
            scheduler.step(loss_sum/sample_num)
        else:
            scheduler.step()

    monitor = {}
    monitor["val_loss"] = loss_sum/sample_num
    monitor["val_accuracy"] = (image_preds_all==image_targets_all).mean()
    return monitor


def inference_one_epoch(model, data_loader, device):
    model.eval()
    image_preds_all = []
    pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    for step, (imgs) in pbar:
        imgs = imgs.to(device).float()

        image_preds = model(imgs)   #output = model(input)
        image_preds_all += [torch.softmax(image_preds, 1).detach().cpu().numpy()]

    image_preds_all = np.concatenate(image_preds_all, axis=0)
    return image_preds_all

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

def inference_one_epoch_tsne(model, data_loader, device):
    model.eval()
    image_preds_all = []
    pbar = tqdm(enumerate(data_loader), total=len(data_loader))

    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook


    for step, (imgs) in pbar:
        imgs = imgs.to(device).float()

        model.model.global_pool.register_forward_hook(get_activation('act2'))
        image_preds = model(imgs)
        image_preds_all += [torch.softmax(activation["act2"], 1).detach().cpu().numpy()]

    image_preds_all = np.concatenate(image_preds_all, axis=0)
    return image_preds_all
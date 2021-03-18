import os
import argparse
import json
import pandas as pd
import os
import torch
from torch import nn

from sklearn.model_selection import StratifiedKFold

from src.augmentation import *
from src.dataset import *
from src.utils import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./config/default.json')

    options = parser.parse_args()
    config = json.load(open(options.config))

    # train 用 df の作成
    train_df = pd.DataFrame()
    base_train_data_path = './flowers-recognition/train/'

    train_data_labels = ['daisy',
                        'dandelion',
                        'rose',
                        'sunflower',
                        'tulip']

    for one_label in train_data_labels:
        one_label_df = pd.DataFrame()
        one_label_paths = os.path.join(base_train_data_path, one_label)
        one_label_df['image_path'] = [os.path.join(one_label_paths, f) for f in os.listdir(one_label_paths)]
        one_label_df['label'] = one_label
        train_df = pd.concat([train_df, one_label_df])
    train_df = train_df.reset_index(drop=True)

    # train の label を数字にエンコードする

    label_dic = {"daisy":0, "dandelion":1, "rose":2,"sunflower":3, "tulip":4}
    train_df["label"]=train_df["label"].map(label_dic)

    # train
    train = train_df
    seed_everything(config['seed'])

    folds = StratifiedKFold(n_splits=config['fold_num'], shuffle=True, random_state=config['seed']).split(np.arange(train.shape[0]), train.label.values)
    for fold, (trn_idx, val_idx) in enumerate(folds):

        if fold > 0: # 時間がかかるので最初のモデルのみ
            break

        print(f'Training with fold {fold} started (train:{len(trn_idx)}, val:{len(val_idx)})')

        train_loader, val_loader = prepare_dataloader(train, (config["img_size_h"], config["img_size_w"]), trn_idx, val_idx, train_bs=config["train_bs"], valid_bs=config["valid_bs"], num_workers=config["num_workers"] )

        device = torch.device(config['device'])

        model = FlowerImgClassifier(config['model_arch'], train.label.nunique(), pretrained=True).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config['T_0'], T_mult=1, eta_min=config['min_lr'], last_epoch=-1)

        loss_tr = nn.CrossEntropyLoss().to(device)
        loss_fn = nn.CrossEntropyLoss().to(device)

        for epoch in range(config['epochs']):
            print(epoch)
            train_one_epoch(epoch, model, loss_tr, optimizer, train_loader, device, config['verbose_step'],scheduler=scheduler, schd_batch_update=False)

            with torch.no_grad():
                valid_one_epoch(epoch, model, loss_fn, val_loader, device, config['verbose_step'], scheduler=None, schd_loss_update=False)

            torch.save(model.state_dict(),f'save/{config["model_arch"]}_fold_{fold}_{epoch}')

        del model, optimizer, train_loader, val_loader,  scheduler
        torch.cuda.empty_cache()
        print("\n")

if __name__ == '__main__':
    main()

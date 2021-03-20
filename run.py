import os
import argparse
import datetime
import logging
import json
import pandas as pd
import os
import torch
from torch import nn

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import  log_loss

from src.augmentation import *
from src.dataset import *
from src.utils import *

def main():
    # config file upload
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./config/default.json')
    options = parser.parse_args()
    config = json.load(open(options.config))

    # log file
    now = datetime.datetime.now()
    logging.basicConfig(
        filename='./logs/log_' + config["model_arch"] + '_'+ '{0:%Y%m%d%H%M%S}.log'.format(now), level=logging.DEBUG
    )
    logging.debug('date : {0:%Y,%m/%d,%H:%M:%S}'.format(now))
    log_list = ["img_size_h", "img_size_w", "train_bs", "monitor"]
    for log_c in log_list:
        logging.debug(f"{log_c} : {config[log_c]}")

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

    if config["psudo"] == "True":
        psudo = pd.read_csv("./flowers-recognition/psudo.csv")
        train_df = pd.concat([train_df, psudo])

    logging.debug(f'psudo : {config["psudo"]}')
    train_df = train_df.reset_index(drop=True)

    # test 用 df の作成
    test_df = pd.DataFrame()
    base_test_data_path = './flowers-recognition/test/'
    test_df['image_path'] = [os.path.join(base_test_data_path, f) for f in os.listdir('./flowers-recognition/test/')]
    test_df = test_df.sort_values('image_path').reset_index(drop=True)

    # train の label を数字にエンコードする

    label_dic = {"daisy":0, "dandelion":1, "rose":2,"sunflower":3, "tulip":4}
    train_df["label"]=train_df["label"].map(label_dic)

    # train
    train = train_df
    seed_everything(config['seed'])
    device = torch.device(config['device'])

    folds = StratifiedKFold(n_splits=config['fold_num'], shuffle=True, random_state=config['seed']).split(np.arange(train.shape[0]), train.label.values)
    for fold, (trn_idx, val_idx) in enumerate(folds):

        if fold > 0: # 時間がかかるので最初のモデルのみ
            break

        print(f'Training with fold {fold} started (train:{len(trn_idx)}, val:{len(val_idx)})')

        train_loader, val_loader = prepare_dataloader(train, (config["img_size_h"], config["img_size_w"]), trn_idx, val_idx, train_bs=config["train_bs"], valid_bs=config["valid_bs"], num_workers=config["num_workers"] )
        model = FlowerImgClassifier(config['model_arch'], train.label.nunique(), pretrained=True).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config['T_0'], T_mult=1, eta_min=config['min_lr'], last_epoch=-1)
        er = EarlyStopping(config['patience'])

        loss_tr = nn.CrossEntropyLoss().to(device)
        loss_fn = nn.CrossEntropyLoss().to(device)

        for epoch in range(config['epochs']):
            train_one_epoch(epoch, model, loss_tr, optimizer, train_loader, device, config['verbose_step'],scheduler=scheduler, schd_batch_update=False)

            with torch.no_grad():
                monitor = valid_one_epoch(epoch, model, loss_fn, val_loader, device, config['verbose_step'], scheduler=None, schd_loss_update=False)

            # Early Stopiing
            if er.update(monitor[config["monitor"]], epoch, config["mode"]) < 0:
                break

            if epoch == er.val_epoch:
                torch.save(model.state_dict(),f'save/{config["model_arch"]}_fold_{fold}_{epoch}')

        del model, optimizer, train_loader, val_loader,  scheduler
        torch.cuda.empty_cache()
        print("\n")

        print("pred start")

    # infer
    train = train_df
    seed_everything(config['seed'])

    folds = StratifiedKFold(n_splits=config['fold_num'], shuffle=True, random_state=config['seed']).split(np.arange(train.shape[0]), train.label.values)


    tst_preds = []
    val_loss = []
    val_acc = []

    # 行数を揃えた空のデータフレームを作成
    cols = ['daisy',
            'dandelion',
            'rose',
            'sunflower',
            'tulip'
           ]

    for fold, (trn_idx, val_idx) in enumerate(folds):
        if fold > 0: # 時間がかかるので最初のモデルのみ
            break

        print(' fold {} started'.format(fold))
        input_shape=(config["img_size_h"], config["img_size_w"])

        valid_ = train.loc[val_idx,:].reset_index(drop=True)
        valid_ds = FlowerDataset(valid_, transforms=get_valid_transforms(input_shape), shape = input_shape, output_label=False)

        test_ds = FlowerDataset(test_df, transforms=get_valid_transforms(input_shape), shape=input_shape, output_label=False)

    val_loader = torch.utils.data.DataLoader(
            valid_ds,
            batch_size=config['valid_bs'],
            num_workers=config['num_workers'],
            shuffle=False,
            pin_memory=False,
        )

    tst_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=config['valid_bs'],
        num_workers=config['num_workers'],
        shuffle=False,
        pin_memory=False,
    )

    device = torch.device(config['device'])
    model = FlowerImgClassifier(config['model_arch'], train.label.nunique()).to(device)

    val_preds = []

    #for epoch in range(config['epochs']-3):
    fold = 0
    model.load_state_dict(torch.load(f'save/{config["model_arch"]}_fold_{fold}_{er.val_epoch}'))
    print(f"used_epoch : {er.val_epoch}")
    logging.debug(f'used_epoch : {er.val_epoch}')
    log_monitor = er.max_val_monitor if config["mode"] == "min" else er.min_val_monitor
    logging.debug(f'val_monitor : {log_monitor}')

    with torch.no_grad():
        val_preds += [inference_one_epoch(model, val_loader, device)]
        tst_preds += [inference_one_epoch(model, tst_loader, device)]

    val_preds = np.mean(val_preds, axis=0)
    val_loss.append(log_loss(valid_.label.values, val_preds))
    val_acc.append((valid_.label.values == np.argmax(val_preds, axis=1)).mean())

    print('validation loss = {:.5f}'.format(np.mean(val_loss)))
    print('validation accuracy = {:.5f}'.format(np.mean(val_acc)))
    tst_preds = np.mean(tst_preds, axis=0)

    del model
    torch.cuda.empty_cache()
    tst_preds_label_all = np.argmax(tst_preds, axis=1)

    # 予測結果を保存
    sub = pd.read_csv("./flowers-recognition/sample_submission.csv")
    sub['class'] = tst_preds_label_all
    label_dic = {0:"daisy", 1:"dandelion", 2:"rose",3:"sunflower", 4:"tulip"}
    sub["class"] = sub["class"].map(label_dic)
    print(sub.value_counts("class"))
    logging.debug(sub.value_counts("class"))
    sub.to_csv(f'output/{config["model_arch"]}_'+ '{0:%Y%m%d%H%M%S}'.format(now) + '_submission.csv', index=False)



if __name__ == '__main__':
    main()

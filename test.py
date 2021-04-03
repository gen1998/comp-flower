from torch import nn
import pandas as pd
import argparse
import json
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

    # infer
    train = train_df
    seed_everything(config['seed'])

    folds = StratifiedKFold(n_splits=config['fold_num'], shuffle=True, random_state=config['seed']).split(np.arange(train.shape[0]), train.label.values)


    tst_preds = []
    val_loss = []
    val_acc = []

    for fold, (trn_idx, val_idx) in enumerate(folds):
        if fold > 0: # 時間がかかるので最初のモデルのみ
            break

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
        model = FlowerImgClassifier(config['model_arch'], train.label.nunique(), config["model_shape"]).to(device)

        val_preds = []

        model.load_state_dict(torch.load(f'save/{config["model_arch"]}_fold_1_7'))

        with torch.no_grad():
            val_preds += [inference_one_epoch(model, val_loader, device)]
            tst_preds += [inference_one_epoch(model, tst_loader, device)]

        val_preds = np.mean(val_preds, axis=0)
        val_loss.append(log_loss(valid_.label.values, val_preds))
        val_acc.append((valid_.label.values == np.argmax(val_preds, axis=1)).mean())
        #print(np.array(tst_preds).shape)



if __name__ == '__main__':
    main()

from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import csv
import numpy as np
import GaussRankScaler
from GaussRankScaler import GaussRankScaler
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from preprocessing import make_preprocessing, make_preprocessing_2, split_train_test
from utils import get_cat_cols, get_cat_cols_from_list, get_not_used

class TrainDataset(Dataset):
    def __init__(self, x, y):
        self.x, self.y = x.astype('float32'), y.astype('float32')

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

class PredictDataset(Dataset):
    def __init__(self, x):
        self.x = x.astype('float32')

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index]


# CSS
def load_css_data(train_data_path,
                  test_data_path=None,
                  target_name='CSS_TARGET'):
    df_train = pd.read_parquet(train_data_path)
    if test_data_path is not None:
        df_test = pd.read_parquet(test_data_path)
    else:
        split_point = int(len(df_train)*0.9)
        df_test = df_train[split_point:]
        df_train = df_train[:split_point]
    with open('../../CSS/feature/css_train_feat_list.csv', mode='r') as csvfile:
        reader = csv.reader(csvfile)
        feat_list = [row for row in reader][0]

    with open('../../CSS/feature/css_categorical_list.csv', mode='r') as csvfile:
        reader = csv.reader(csvfile)
        cat_cols = []#[row for row in reader][0]

    num_cols = list(set(feat_list) - set(cat_cols))

    X_num = np.vstack([df_train[num_cols].to_numpy(),
                       df_test[num_cols].to_numpy()])
    if len(cat_cols) == 0:
        X = X_num
        X_cat = np.zeros(shape=(0,)).reshape(1, -1)
    else:
        X_cat = np.vstack([df_train[cat_cols].to_numpy(),
                           df_test[cat_cols].to_numpy()])
        encoder = OneHotEncoder(sparse=False)
        X_cat = encoder.fit_transform(X_cat)
        X = np.hstack([X_cat, X_num])
    y = pd.concat([df_train[target_name], df_test[target_name]], axis=0).to_numpy().reshape(-1, 1)

    return X, y, len(df_train), X_cat.shape[1], X_num.shape[1]

# Kaggle (https://www.kaggle.com/competitions/amex-default-prediction)
def load_data():
    X_train_total, X_train_label, X_valid_label = _get_data_from_csv()
    X = pd.read_csv('data/X_data_2.csv').to_numpy()
    y = pd.read_csv('data/y_data_2.csv').to_numpy()
    X_train = pd.read_csv('data/X_train_2.csv')

    not_used_cols = list(set(X_train.columns) & set(get_not_used()))
    cat_cols = get_cat_cols_from_list(X_train.columns)
    num_cols = list(set(X_train.columns) - set(cat_cols) - set(not_used_cols))

    return X, y, len(X_train_label), X.shape[1] - len(num_cols), len(num_cols)

def get_data():
    X_train_total, X_train_label, X_valid_label = _get_data_from_csv()

    X_train = make_preprocessing_2(X_train_total)
    df_x_train, df_x_test = split_train_test(X_train, X_train_label, X_valid_label, False)

    df_x_train = df_x_train.fillna(-1)
    df_x_train = df_x_train.replace([np.inf, -np.inf], -1)

    df_x_test = df_x_test.fillna(-1)
    df_x_test = df_x_test.replace([np.inf, -np.inf], -1)

    not_used_cols = list(set(X_train.columns) & set(get_not_used()))
    cat_cols = get_cat_cols_from_list(X_train.columns)
    num_cols = list(set(X_train.columns) - set(cat_cols) - set(not_used_cols))

    # gauss rank scaler
    gs = GaussRankScaler()
    gs.fit(df_x_train[num_cols])
    df_x_train[num_cols] = pd.DataFrame(gs.transform(df_x_train[num_cols]), columns=num_cols)
    df_x_test[num_cols] = pd.DataFrame(gs.transform(df_x_test[num_cols]), columns=num_cols)
    # standard scaler
    # ss = StandardScaler()
    # ss.fit(df_x_train[num_cols])
    # df_x_train[num_cols] = pd.DataFrame(ss.transform(df_x_train[num_cols]), columns=num_cols)
    # df_x_test[num_cols] = pd.DataFrame(ss.transform(df_x_test[num_cols]), columns=num_cols)

    X_num = np.vstack([df_x_train[num_cols].to_numpy(),
                       df_x_test[num_cols].to_numpy()])

    X_cat = np.vstack([df_x_train[cat_cols].to_numpy(),
                       df_x_test[cat_cols].to_numpy()])

    encoder = OneHotEncoder(sparse=False)
    X_cat = encoder.fit_transform(X_cat)

    X = np.hstack([X_cat, X_num])
    y = pd.concat([df_x_train['target'], df_x_test['target']], axis=0).to_numpy().reshape(-1, 1)

    pd.DataFrame(X).to_csv('data/X_data_2.csv', header=None, index=None)
    pd.DataFrame(y).to_csv('data/y_data_2.csv', header=None, index=None)
    X_train.to_csv('data/X_train_2.csv', index=None)
    return X, y, len(X_train_label), X_cat.shape[1], X_num.shape[1]

def _get_data_from_csv():
    # total_data
    # base_dir = '/data/kaggle/data/'
    # X_train_total = pd.read_csv(os.path.join(base_dir, 'train_data.csv'))

    # splited data
    base_dir = '/data/kaggle/data2'
    X_train_label = pd.read_parquet(os.path.join(base_dir, 'train_label.parquet'))
    X_valid_label = pd.read_parquet(os.path.join(base_dir, 'valid_label.parquet'))
    # X_total_label = pd.concat([X_train_label, X_valid_label], axis=0).reset_index(drop=True)

    # int data
    X_train_int = pd.read_parquet(os.path.join(base_dir, 'X_train_int.parquet'))
    # X_test_int = pd.read_parquet(os.path.join(base_dir, 'X_test_int.parquet'))

    return X_train_int, X_train_label, X_valid_label


# return dataloader for DAE
def get_dataloader_for_mlp(X,
                           y,
                           mlp_batch_size,
                           cut_off_valid,
                           len_train):
    train_dl = DataLoader(dataset=TrainDataset(X[:cut_off_valid], y[:cut_off_valid]),
                          batch_size=mlp_batch_size,
                          shuffle=True,
                          num_workers=64,
                          pin_memory=True,
                          drop_last=True)

    valid_dl = DataLoader(dataset=TrainDataset(X[cut_off_valid:len_train], y[cut_off_valid:len_train]),
                          batch_size=mlp_batch_size,
                          shuffle=False,
                          num_workers=32,
                          pin_memory=True,
                          drop_last=False)

    test_dl = DataLoader(dataset=PredictDataset(X[len_train:]),
                         batch_size=mlp_batch_size,
                         shuffle=False,
                         pin_memory=True,
                         drop_last=False)
    return train_dl, valid_dl, test_dl

def get_dataloader_for_dae(X,
                           dae_batch_size):
    dae_dl = DataLoader(dataset=PredictDataset(X),
                        batch_size=dae_batch_size,
                        num_workers=64,
                        shuffle=True,
                        pin_memory=True,
                        drop_last=True)

    return dae_dl
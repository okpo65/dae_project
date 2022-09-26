import argparse
import seaborn as sns
from pylab import rcParams
from utils import str_to_bool

import numpy as np
import pandas as pd
import os
import torch
from tqdm import tqdm

from dataset import get_data, load_data, load_css_data, get_dataloader_for_dae, get_dataloader_for_mlp
from metric import amex_metric, CSSTask
from config import DAE_CFG, MLP_CFG, TRANSFORMER_DAE_CFG, WANDB_KEY
from train import train_mlp, train_tree_model, train_dae, train_transformer_dae
from test import test_mlp, test_tree_model
from load import load_mlp_model, load_deepstack_dae_model, load_transformer_dae_model, load_bottleneck_dae_model
from time import gmtime, strftime
from datetime import datetime
from pytz import timezone
from enum_container import DAEModelType, MetaModelType, TaskType
import wandb

# parser
parser = argparse.ArgumentParser()
parser.add_argument('--task_type', type=TaskType.from_string, default='css')
parser.add_argument('--train_data_path', type=str)
parser.add_argument('--test_data_path', type=str, default=None)
parser.add_argument('--dae_model_type', type=DAEModelType.from_string, default='DeepStack')
parser.add_argument('--dae_batch_size', type=int, default=512)
parser.add_argument('--dae_model_path', type=str, default='')
parser.add_argument('--meta_model_path', type=str, default='')
parser.add_argument('--meta_model_type', type=MetaModelType.from_string, default='MLP')
parser.add_argument('--refresh_input',type=int, default=0, help='decide to load input data or get a saved data')
parser.add_argument('--train_dae_model', type=int, default=1, help='decide to train DAE')
parser.add_argument('--train_meta_model', type=int, default=1, help='decide to train MLP')

def main():
    opt = parser.parse_args()
    # wandb
    if opt.train_dae_model == 1 or (opt.train_meta_model == 1 and opt.meta_model_type == MetaModelType.MLP):
        wandb.login(key=WANDB_KEY)
        wandb_obj = wandb.init(project=WANDB_KEY, reinit=True)

    print('-'*30)
    print('start load data')
    print('-'*30)

    if opt.task_type == TaskType.css:
        X, y, len_train, len_cat, len_num = load_css_data(opt.train_data_path,
                                                          opt.test_data_path,
                                                          'target')
    elif opt.task_type == TaskType.kaggle:
        if opt.refresh_input == 1:
            X, y, len_train, len_cat, len_num = get_data()
        else:
            X, y, len_train, len_cat, len_num = load_data()
    print('-'*30)
    print('end load data')
    print('-'*30)

    print(len(X), len(y), len_train, len_cat, len_num)

    print('-' * 30)
    print('start dae training')
    print('-' * 30)
    # DAE Training
    dae_dl = get_dataloader_for_dae(X,
                                    opt.dae_batch_size)
    if opt.dae_model_type == DAEModelType.DeepStack:
        if opt.train_dae_model == 1:
            dae = train_dae(load_deepstack_dae_model(len_cat, len_num),
                            dae_dl,
                            len_cat,
                            len_num)
        else:
            dae = load_deepstack_dae_model(len_cat,
                                           len_num,
                                           opt.dae_model_path)
        dae_hidden_size = DAE_CFG['hidden_size']
    elif opt.dae_model_type == DAEModelType.DeepBottleneck:
        if opt.train_dae_model == 1:
            dae = train_dae(load_bottleneck_dae_model(len_cat, len_num),
                            dae_dl,
                            len_cat,
                            len_num)
        else:
            dae = load_bottleneck_dae_model(len_cat,
                                           len_num,
                                           opt.dae_model_path)
        dae_hidden_size = DAE_CFG['hidden_size']
    elif opt.dae_model_type == DAEModelType.TransformerAutoEncoder:
        if opt.train_dae_model == 1:
            dae = train_transformer_dae(dae_dl,
                                        len_cat,
                                        len_num)
        else:
            dae = load_transformer_dae_model(len_cat,
                                             len_num,
                                             opt.dae_model_path)
        dae_hidden_size = TRANSFORMER_DAE_CFG['hidden_size']

    print('-' * 30)
    print('start meta-model training')
    print('-' * 30)
    # # saving DAE result
    # with torch.no_grad():
    #     dae_x = dae.feature(torch.Tensor(X).to(torch.device('cuda')))
    #     dae_x = dae_x.detach().cpu().numpy()
    #
    # dae_x = pd.DataFrame(dae_x)
    # dae_x.columns = ['col' + f'{idx}' for idx in dae_x.columns.tolist()]
    # dae_x['target'] = y.reshape(1, -1)[0]
    # print('sdfsdfsf',dae_x)
    # dae_x.to_parquet('../../CSS/dataset/X_jan_dae_result.parquet')
    # return
    # Meta-model Training
    cut_off_valid = int(len_train * 0.9)
    train_dl, valid_dl, test_dl = get_dataloader_for_mlp(X,
                                                         y,
                                                         mlp_batch_size=MLP_CFG['batch_size'],
                                                         cut_off_valid=cut_off_valid,
                                                         len_train=len_train)

    target_name = 'CSS_TARGET' if opt.task_type == TaskType.css else 'target'
    y_true = pd.DataFrame({target_name: y.reshape(1, -1)[0]}).loc[len_train:].reset_index(drop=True)
    if opt.meta_model_type == MetaModelType.MLP:
        if opt.train_meta_model == 1:
            wandb_obj.finish()
            wandb.init()
            mlp = train_mlp(train_dl,
                            valid_dl,
                            dae,
                            dae_hidden_size,
                            3)
        else:
            mlp = load_mlp_model(opt.meta_model_path,
                                 dae_hidden_size,
                                 3)
        predictions = test_mlp(dae,
                               mlp,
                               test_dl)
    else:
        tree_model_list = train_tree_model(dae,
                                           train_dl,
                                           valid_dl,
                                           opt.meta_model_type,
                                           1,
                                           target_name)
        predictions, raw_predictions = test_tree_model(tree_model_list, dae, test_dl)
    if opt.task_type == TaskType.css:
        task = CSSTask()
        if 'raw_predictions' in locals():
            print('-' * 30)
            for idx, raw_pred in enumerate(raw_predictions):
                result = task.eval_seg_ks(raw_pred, y_true)
                print(f'{opt.task_type}: ensemble fold {idx} result: {result}')
            print('-' * 30)
        task.draw_seg_ks(predictions, y_true)
        print('sdfsdfs', predictions, y_true)
        result = task.eval_seg_ks(predictions, y_true)

    elif opt.task_type == TaskType.kaggle:
        predictions = pd.DataFrame({'prediction': predictions})
        result = amex_metric(y_true, predictions)
    print('-'*30)
    print(f'{opt.task_type}: result: {result}')
    print('-'*30)

    # save result
    save_dir = os.path.join('results/', f"{datetime.now(timezone('Asia/Seoul')).strftime('%Y-%m-%d')}")
    cur_time = datetime.now(timezone('Asia/Seoul')).strftime('%H:%M')
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    pd.DataFrame(predictions).to_csv(f"{save_dir}/{opt.task_type}_{opt.dae_model_type}_{opt.meta_model_type}_{cur_time}.csv", header=None, index=None)
    y_true.to_csv(f"{save_dir}/{opt.task_type}_{opt.dae_model_type}_{opt.meta_model_type}_{cur_time}_true.csv", header=None, index=None)

if __name__ == "__main__":
    main()
import torch
import os
import numpy as np
import pandas as pd
from config import TRANSFORMER_DAE_CFG, DAE_CFG, MLP_CFG, TRANSFORMER_DAE_TRAIN_CFG
from DAE import DeepStackDAE, TransformerAutoEncoder
from MLP import MLPModel
from TreeModels import CatBoostClassiferModel, LGBMClassifierModel
from tqdm import tqdm
from utils import EarlyStopping, AverageMeter, SwapNoiseMasker
from loss import FocalLoss
from sklearn.metrics import mean_squared_error
from time import gmtime, strftime
from datetime import datetime
from pytz import timezone
from sklearn.model_selection import StratifiedKFold
from enum_container import MetaModelType
import copy
import wandb

def train_transformer_dae(dae_dl,
                          len_cat,
                          len_num):

    transformer_dae = TransformerAutoEncoder(
        num_inputs=len_cat+len_num,
        n_cats=len_cat,
        n_nums=len_num,
        **TRANSFORMER_DAE_CFG
    ).cuda()

    repeats = [len_cat, len_num]
    probas = [.25, .25]
    swap_probas = sum([[p] * r for p, r in zip(probas, repeats)], [])

    noise_maker = SwapNoiseMasker(swap_probas)
    optimizer = torch.optim.Adam(transformer_dae.parameters(), lr= TRANSFORMER_DAE_TRAIN_CFG['init_lr'])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=TRANSFORMER_DAE_TRAIN_CFG['lr_decay'])
    model_dir = os.path.join('save_model/DAE_model/', f'{strftime("%Y-%m-%d", gmtime())}')
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    # train model
    for epoch in tqdm(range(TRANSFORMER_DAE_TRAIN_CFG['max_epochs'])):
        transformer_dae.train()
        meter = AverageMeter()
        for i, x in enumerate(dae_dl):
            x = x.cuda()
            x_corrputed, mask = noise_maker.apply(x)
            optimizer.zero_grad()
            loss = transformer_dae.loss(x_corrputed, x, mask)
            loss.backward()
            wandb.log({'transformer_dae_loss': loss})
            optimizer.step()

            meter.update(loss.detach().cpu().numpy())

        scheduler.step()
        if epoch % 10 == 0:
            print('epoch {:5d} - loss {:.6f}'.format(epoch, meter.avg))
        if epoch % 100 == 0:
            torch.save({
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "model": transformer_dae.state_dict()
            }, f'{model_dir}/{epoch}_transformer_dae_checkpoint.pth')

def train_dae(dae,
              dae_dl,
              len_cat,
              len_num):
    optimizer = torch.optim.Adam(
        dae.parameters(),
        lr=DAE_CFG['init_lr']
    )

    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=DAE_CFG['lr_gamma']
    )
    earlystopper = EarlyStopping(mode='min',
                                 min_delta=1e-7,
                                 patience=200,
                                 percentage=False,
                                 verbose=1)

    repeats = [len_cat, len_num]
    probas = [.8, DAE_CFG['noise_ratio']]
    swap_probas = sum([[p] * r for p, r in zip(probas, repeats)], [])

    noise_maker = SwapNoiseMasker(swap_probas)

    print('-------------------------------------------')
    print('start DAE learning')
    print('-------------------------------------------')

    model_dir = os.path.join('save_model/DAE_model/', f"{datetime.now(timezone('Asia/Seoul')).strftime('%Y-%m-%d')}")
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    wandb.watch(dae)
    for epoch in tqdm(range(3001)):
        dae.train()
        meter = AverageMeter()
        for i, x in enumerate(dae_dl):
            x = x.cuda()
            noisy_x, mask = noise_maker.apply(x)
            optimizer.zero_grad()
            loss = dae.loss(noisy_x,
                            x,
                            mask,
                            weights=[10, 14])
            loss.backward()
            optimizer.step()
            meter.update(loss.detach().cpu().numpy())
            wandb.log({'loss': loss})
        scheduler.step()
        if epoch % 100 == 0:
            print(epoch, meter.avg)
            model_checkpoint = f"{model_dir}/{epoch}_{datetime.now(timezone('Asia/Seoul')).strftime('%H:%M')}_dae_checkpoint.pth"
            torch.save({
                "epoch": epoch,
                "loss": loss,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "model": dae.state_dict()
            }, model_checkpoint)
        if epoch % 10 == 0:
            print(f'epoch: {epoch}, loss: {loss.detach().cpu().numpy()}')
        if earlystopper.step(meter.avg): break

    print('-------------------------------------------')
    print('end DAE learning')
    print('-------------------------------------------')
    return dae


def train_mlp(train_dl,
              valid_dl,
              dae,
              dae_hidden_size,
              num_hidden_size):
    model = MLPModel(
        50,
        hidden_size=MLP_CFG['hidden_size'],
        input_dropout=MLP_CFG['input_dropout'],
        dropout_rate=MLP_CFG['dropout']
    ).cuda()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=MLP_CFG['init_lr'],
        weight_decay=MLP_CFG['l2_reg']
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min',
                                                           factor=1 / 3,
                                                           patience=20,
                                                           verbose=0,
                                                           cooldown=2,
                                                           min_lr=1e-7)

    earlystopper = EarlyStopping(mode='min',
                                 min_delta=1e-7,
                                 patience=30,
                                 percentage=False,
                                 verbose=1)

    model_dir = os.path.join('save_model/MLP_model/', f"{datetime.now(timezone('Asia/Seoul')).strftime('%Y-%m-%d')}")
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    wandb.watch(model)

    for epoch in range(777):
        model.train()
        for i, (x, target) in enumerate(train_dl):
            x, target = x.cuda(), target.cuda()
            with torch.no_grad():
                x = dae.feature(x)
            optimizer.zero_grad()
            # loss = FocalLoss()(model.forward(x), target)
            loss = torch.nn.functional.binary_cross_entropy(model.forward(x), target)
            loss.backward()
            wandb.log({'loss_mlp': loss})
            optimizer.step()

        model.eval()
        predictions = []
        with torch.no_grad():
            for _, (x, _) in enumerate(valid_dl):
                x = dae.feature(x.cuda())
                prediction = model.forward(x)
                predictions.append(prediction.detach().cpu().numpy())
        predictions = np.concatenate(predictions)
        # valid_loss = FocalLoss(torch.from_numpy(predictions), torch.Tensor(valid_dl.dataset.y), reduction='none')
        valid_loss = torch.nn.functional.binary_cross_entropy(torch.from_numpy(predictions), torch.Tensor(valid_dl.dataset.y))
        valid_rmse = mean_squared_error(valid_dl.dataset.y, predictions, squared=False)
        wandb.log({'valid_rmse': valid_rmse})
        wandb.log({'valid_loss': valid_loss})
        scheduler.step(valid_loss)
        if epoch % 20 == 0:
            print(epoch, valid_loss)
            model_checkpoint = f"{model_dir}/{epoch}_{datetime.now(timezone('Asia/Seoul')).strftime('%H:%M')}_mlp_checkpoint.pth"
            torch.save({
                "epoch": epoch,
                "loss": loss,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "model": model.state_dict()
            }, model_checkpoint)

        if earlystopper.step(valid_loss): break
    print('-------------------------------------------')
    print('end MLP learning')
    print('-------------------------------------------')
    return model

def train_tree_model(dae,
                     train_dl,
                     valid_dl,
                     model_type,
                     n_splits=5,
                     target_name='CSS_TARGET'):
    dae.eval()
    X = np.vstack([train_dl.dataset.x, valid_dl.dataset.x])
    y = np.vstack([train_dl.dataset.y, valid_dl.dataset.y])

    with torch.no_grad():
        dae_x = dae.feature(torch.Tensor(X).to(torch.device('cuda')))
        dae_x = dae_x.detach().cpu().numpy()
    if n_splits == 1:
        len_train = len(train_dl.dataset.x)
        cut_off_valid = len_train - len(valid_dl.dataset.x)
    else:
        len_train = len(train_dl.dataset.x) + len(valid_dl.dataset.x)
        cut_off_valid = len_train

    x_train, y_train = dae_x[:cut_off_valid], y[:cut_off_valid]
    x_valid, y_valid = dae_x[cut_off_valid:len_train], y[cut_off_valid:len_train]

    x_train = pd.DataFrame(x_train)
    y_train = pd.DataFrame({target_name: y_train.reshape(1, -1)[0]})

    model_list = []
    if model_type == MetaModelType.Catboost:
        model = CatBoostClassiferModel()
    elif model_type == MetaModelType.LGBM:
        model = LGBMClassifierModel()
    if n_splits == 1:
        model.fit_model(x_train, y_train, x_valid, y_valid)
        model_list.append(model)
    # ensemble
    else:
        model_list = _train_ensemble_tree_model(x_train,
                                                y_train,
                                                n_splits,
                                                model,
                                                target_name)

    print('model_list21421',model_list)
    return model_list

def _train_ensemble_tree_model(x_train,
                               y_train,
                               n_splits,
                               model,
                               target_name='CSS_TARGET'):
    # Create a numpy array to store out of folds predictions
    oof_predictions = np.zeros(len(x_train))
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    model_list = []
    for fold, (trn_ind, val_ind) in enumerate(kfold.split(x_train, y_train[target_name])):
        print(' ')
        print('-' * 50)
        print(f'Training fold {fold} with {x_train.shape} features...')
        _model = copy.deepcopy(model)
        x_fold_train, x_fold_val = x_train.iloc[trn_ind], x_train.iloc[val_ind]
        y_fold_train, y_fold_val = y_train[target_name].iloc[trn_ind], y_train[target_name].iloc[val_ind]
        _model.fit_model(x_fold_train, y_fold_train, x_fold_val, y_fold_val)
        # Predict validation
        val_pred = _model.get_model().predict_proba(x_fold_val)
        val_pred = val_pred[:, 1]
        # Add to out of folds array
        oof_predictions[val_ind] = val_pred
        # Predict the test set
        model_list.append(_model)
        # verbose
    return model_list
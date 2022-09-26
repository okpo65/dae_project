import torch
import numpy as np
import pandas as pd
from operator import lt, gt

class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val, self.avg, self.sum, self.count = 0,0,0,0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, percentage=False, patience=10, initial_bad=0, initial_best=np.nan, verbose=0):
        self.mode = mode
        self.patience = patience
        self.best = float('inf') if mode == 'min' else float('-inf')
        self.num_bad_epochs = initial_bad
        self.is_better = self._init_is_better(mode, min_delta, percentage)
        self.verbose = verbose
        self._stop = False

    def step(self, metric):
        if self.is_better(metric, self.best):
            self.num_bad_epochs = 0
            self.best = metric
        else:
            self.num_bad_epochs += 1

        if np.isnan(self.best) and (not np.isnan(metric)):
            self.num_bad_epochs = 0
            self.best = metric

        self._stop = self.num_bad_epochs >= self.patience
        if self.verbose and self._stop: print('Early Stopping Triggered, best score is: ', self.best)
        return self._stop

    def _init_is_better(self, mode, min_delta, percentage):
        comparator = lt if mode == 'min' else gt
        if not percentage:
            def _is_better(new, best):
                target = best - min_delta if mode == 'min' else best + min_delta
                return comparator(new, target)
        else:
            def _is_better(new, best):
                target = best * (1 - (min_delta / 100)) if mode == 'min' else best * (1 + (min_delta / 100))
                return comparator(new, target)
        return _is_better

class SwapNoiseMasker(object):
    def __init__(self, probas):
        self.probas = torch.from_numpy(np.array(probas))

    def apply(self, X):
        should_swap = torch.bernoulli(self.probas.to(X.device) * torch.ones((X.shape)).to(X.device))
        corrupted_X = torch.where(should_swap == 1, X[torch.randperm(X.shape[0])], X)
        mask = (corrupted_X != X).float()
        return corrupted_X, mask


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='none'):
        super(FocalLoss, self).__init__()
        self.loss_fn = torch.nn.BCEWithLogitsLoss(reduction=reduction)
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = self.loss_fn.reduction  # mean, sum, etc..

    def forward(self, pred, true):
        bceloss = self.loss_fn(pred, true)

        pred_prob = torch.sigmoid(pred)  # p  pt는 p가 true 이면 pt = p / false 이면 pt = 1 - p
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)  # add balance
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma  # focal term
        loss = alpha_factor * modulating_factor * bceloss  # bceloss에 이미 음수가 들어가 있음

        if self.reduction == 'mean':
            return loss.mean()

        elif self.reduction == 'sum':
            return loss.sum()

        else:  # 'none'
            return loss


def add_period_aggregation(df, stats, cols=[], sequence_list=[], reverse=False):
    _df = df.copy()
    if 'sequence' not in _df.columns:
        _df['sequence'] = _df.sort_values(by='S_2', ascending=False).groupby('customer_ID').cumcount() + 1
    for idx, sequence in enumerate(sequence_list):
        if reverse:
            tmp_df = _df[_df['sequence'] >= sequence]
            seq_name = f'_reverse_{sequence}'
        else:
            tmp_df = _df[_df['sequence'] <= sequence]
            seq_name = f'_{sequence}'

        use_cols = list(set(cols) - set(get_not_used()))
        num_aggregations = {}
        for col in use_cols:
            num_aggregations[col] = stats
        tmp_df_agg = tmp_df.groupby('customer_ID').agg({**num_aggregations})
        tmp_df_agg.columns = pd.Index([e[0] + "_" + e[1] + seq_name for e in tmp_df_agg.columns.tolist()])
        tmp_df_agg.reset_index(inplace=True)
        tmp_df_agg.fillna(value=0, inplace=True)

        if idx == 0:
            df_agg = tmp_df_agg
        else:
            df_agg = df_agg.merge(tmp_df_agg, on='customer_ID')
    return df_agg

# about feature list
def get_not_used():
    # cid is the label encode of customer_ID
    # row_id indicates the order of rows
    return ['id', 'customer_ID', 'target', 'cid', 'S_2', 'date', 'sequence', 'S_2_x']

def get_cat_cols():
    return ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']

def get_cat_cols_from_list(col_list):
    res = []
    for col in col_list:
        for cat_col in get_cat_cols():
            if cat_col in col and 'encode' not in col:
                res.append(col)
                break
    return res

# Common
def str_to_bool(value):
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')

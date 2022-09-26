import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import roc_curve as r_curve
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score

# CSS

class SegMetric:
    def __init__(self, task='css'):
        self.task = task
        self.num = 100

    def ks(self, default_prob, y_true, ax, name):
        assert default_prob.shape == y_true.shape
        restable = pd.DataFrame({'default': y_true})
        restable['nondefault'] = 1 - restable['default']
        restable['prob1'] = default_prob
        restable['prob0'] = 1 - default_prob
        restable['bucket'] = pd.qcut(restable['prob0'], self.num, duplicates='drop')
        grouped = restable.groupby('bucket', as_index=False)
        kstable = pd.DataFrame()
        kstable['min_prob'] = grouped.min()['prob0']
        kstable['max_prob'] = grouped.max()['prob0']
        kstable['defaults'] = grouped.sum()['default']
        kstable['nondefaults'] = grouped.sum()['nondefault']
        kstable = kstable.sort_values(by='min_prob', ascending=True).reset_index(drop=True)
        kstable['default_rate'] = (kstable.defaults / restable['default'].sum()).apply('{0:.2%}'.format)
        kstable['nondefault_rate'] = (kstable.nondefaults / restable['nondefault'].sum()).apply('{0:.2%}'.format)
        kstable['cum_defaultrate'] = (kstable.defaults / restable['default'].sum()).cumsum()
        kstable['cum_nondefaultrate'] = (kstable.nondefaults / restable['nondefault'].sum()).cumsum()
        kstable['KS'] = np.round(kstable['cum_defaultrate'] - kstable['cum_nondefaultrate'], 3) * 100

        def draw_max_ks(x):
            ax.plot([x.min_prob, x.min_prob], [x.cum_nondefaultrate, x.cum_defaultrate], color='#FA1600')
            print(f'[{name}] KS is {x.KS}% at score {x.min_prob}')

        max_row = kstable.loc[kstable['KS'] == max(kstable['KS']), :]
        max_row = max_row.copy()
        max_row['min_prob'] = max_row['min_prob'].map(lambda x: round(x, 2))
        max_row = max_row.drop_duplicates(['min_prob'])
        max_row.head(1).apply(draw_max_ks, axis=1)
        sns.lineplot(x=kstable['min_prob'], y=kstable.cum_defaultrate, label='Default', color='#000000', ax=ax)
        sns.lineplot(x=kstable['min_prob'], y=kstable.cum_nondefaultrate, label='Non-Default', color='#265BFA', ax=ax)
        ax.set_ylabel('Cumulative rate')
        ax.set_xlabel('Score')
        ax.set(ylim=(0, 1))
        ax.set_title('[' + name + ']: KS Score')
        kstable['cum_defaultrate'] = kstable['cum_defaultrate'].apply('{0:.2%}'.format)
        kstable['cum_nondefaultrate'] = kstable['cum_nondefaultrate'].apply('{0:.2%}'.format)
        return kstable

    def get_ks_table(self, default_prob, y_true):
        restable = self.get_res_table(default_prob, y_true)
        grouped = restable.groupby('bucket', as_index=False)
        kstable = pd.DataFrame()
        kstable['min_prob'] = grouped.min()['prob0']
        kstable['max_prob'] = grouped.max()['prob0']
        kstable['defaults'] = grouped.sum()['default']
        kstable['nondefaults'] = grouped.sum()['nondefault']
        kstable = kstable.sort_values(by='min_prob', ascending=True).reset_index(drop=True)
        kstable['default_rate'] = (kstable.defaults / restable['default'].sum()).apply('{0:.2%}'.format)
        kstable['nondefault_rate'] = (kstable.nondefaults / restable['nondefault'].sum()).apply('{0:.2%}'.format)
        kstable['cum_defaultrate'] = (kstable.defaults / restable['default'].sum()).cumsum()
        kstable['cum_nondefaultrate'] = (kstable.nondefaults / restable['nondefault'].sum()).cumsum()
        kstable['KS'] = np.round(kstable['cum_defaultrate'] - kstable['cum_nondefaultrate'], 3) * 100
        #         kstable['cum_defaultrate'] = kstable['cum_defaultrate'].apply('{0:.2%}'.format)
        #         kstable['cum_nondefaultrate'] = kstable['cum_nondefaultrate'].apply('{0:.2%}'.format)
        return kstable

    def get_res_table(self, default_prob, y_true):
        assert default_prob.shape == y_true.shape
        restable = pd.DataFrame({'default': y_true})
        restable['nondefault'] = 1 - restable['default']
        restable['prob1'] = default_prob
        restable['prob0'] = 1 - default_prob
        restable['bucket'] = pd.qcut(restable['prob0'], self.num, duplicates='drop')

        return restable

    def get_max_ks(self, default_prob, y_true):
        kstable = self.get_ks_table(default_prob, y_true)
        max_row = kstable.loc[kstable['KS'] == max(kstable['KS']), :]
        max_row = max_row.copy()
        max_row['min_prob'] = max_row['min_prob'].map(lambda x: round(x, 2))
        return max_row.KS.head(1).values[0]

    def roc_curve(self, y_pred, y_true, ax, name, pos_label=1):
        assert y_true.shape == y_pred.shape
        fpr, tpr, thresholds = r_curve(y_true, y_pred, pos_label=pos_label)
        roc_auc = metrics.auc(fpr, tpr)
        display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=self.task)
        display.plot(ax=ax)
        ax.set_title('[' + name + ']: ROC Curve')
        print(f'[{name}] ROC_AUC : {roc_auc}')

    def get_roc_curve(self, default_prob, y_true, pos_label=1):
        #         y_true = y_true['CSS_TARGET']

        fpr, tpr, thresholds = r_curve(y_true, default_prob, pos_label=pos_label)
        roc_auc = metrics.auc(fpr, tpr)
        return roc_auc

    def draw_ks(self, default_prob, y_true, ax, name):
        def draw_max_ks(x):
            ax.plot([x.min_prob, x.min_prob], [x.cum_nondefaultrate, x.cum_defaultrate], color='#FA1600')

        kstable = self.get_ks_table(default_prob, y_true)
        max_row = kstable.loc[kstable['KS'] == max(kstable['KS']), :]
        max_row = max_row.copy()
        max_row['min_prob'] = max_row['min_prob'].map(lambda x: round(x, 2))

        max_row = max_row.drop_duplicates(['min_prob'])
        max_row.head(1).apply(draw_max_ks, axis=1)
        sns.lineplot(x=kstable['min_prob'], y=kstable.cum_defaultrate, label='Default', color='#000000', ax=ax)
        sns.lineplot(x=kstable['min_prob'], y=kstable.cum_nondefaultrate, label='Non-Default', color='#265BFA', ax=ax)
        ax.set_ylabel('Cumulative rate')
        ax.set_xlabel('Score')
        ax.set(ylim=(0, 1))
        ax.set_title('[' + name + ']: KS Score')

class CSSTask:
    def eval_seg_ks(self, score, y_test):
        y_test = y_test['CSS_TARGET']
        if score.shape != y_test.shape:
            if score.shape[1] == 2:
                score = score[:, 1]
            elif score.shape[1] == 1:
                score = score.reshape(-1)
        assert score.shape == y_test.shape
        df = pd.DataFrame({'score': score})
        metric = SegMetric(task='css')
        return (metric.get_max_ks(df['score'], y_test),
                metric.get_roc_curve(df['score'], y_test))

    def draw_seg_ks(self, score, y_test, name="all"):
        y_test = y_test['CSS_TARGET']
        if score.shape != y_test.shape:
            if score.shape[1] == 2:
                score = score[:, 1]
            elif score.shape[1] == 1:
                score = score.reshape(-1)
        assert score.shape == y_test.shape
        df = pd.DataFrame({'score': score})
        metric = SegMetric(task='css')
        fig, axes = plt.subplots(ncols=1, figsize=(6, 6))
        metric.draw_ks(df['score'], y_test, axes, name)
        fig.savefig('results/KS_fig.png')

# Kaggle
def amex_metric(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
    def top_four_percent_captured(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = (pd.concat([y_true, y_pred], axis='columns').sort_values('prediction', ascending=False))
        df['weight'] = df['target'].apply(lambda x: 20 if x == 0 else 1)
        four_pct_cutoff = int(0.04 * df['weight'].sum())
        df['weight_cumsum'] = df['weight'].cumsum()
        df_cutoff = df.loc[df['weight_cumsum'] <= four_pct_cutoff]
        return (df_cutoff['target'] == 1).sum() / (df['target'] == 1).sum()

    def weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = (pd.concat([y_true, y_pred], axis='columns').sort_values('prediction', ascending=False))
        df['weight'] = df['target'].apply(lambda x: 20 if x == 0 else 1)
        df['random'] = (df['weight'] / df['weight'].sum()).cumsum()
        total_pos = (df['target'] * df['weight']).sum()
        df['cum_pos_found'] = (df['target'] * df['weight']).cumsum()
        df['lorentz'] = df['cum_pos_found'] / total_pos
        df['gini'] = (df['lorentz'] - df['random']) * df['weight']
        return df['gini'].sum()

    def normalized_weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        y_true_pred = y_true.rename(columns={'target': 'prediction'})
        return weighted_gini(y_true, y_pred) / weighted_gini(y_true, y_true_pred)
    g = normalized_weighted_gini(y_true, y_pred)
    d = top_four_percent_captured(y_true, y_pred)

    return 0.5 * (g + d)
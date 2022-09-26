from utils import get_not_used, get_cat_cols, get_cat_cols_from_list, add_period_aggregation
from sklearn.preprocessing import LabelEncoder
import pandas as pd
# CSS

# Kaggle (https://www.kaggle.com/competitions/amex-default-prediction)
def make_preprocessing(_df):
    df = _df.copy()
    # columns
    num_cols = list(set(df.columns) - set(get_not_used()) - set(get_cat_cols()))
    cat_cols = get_cat_cols()
    # add sequence
    if 'sequence' not in _df.columns:
        _df = add_sequence(_df)
        df = _df.copy()

    # num agg
    df_stat = add_period_aggregation(df, ['mean', 'std', 'last'], num_cols, [13])
    df1 = df_stat

    # cat agg
    df_stat_cat = add_period_aggregation(df, ['last'], cat_cols, [13])
    df1 = df1.merge(df_stat_cat, on='customer_ID')

    # label encoding on cat features (overwrite)
    cat_cols = get_cat_cols_from_list(df1.columns.tolist())
    df1 = encode_cat_label(df1, cat_cols)

    # S_2 diff feature
    df_s2_diff = add_s2_diff(df)
    df1 = df1.merge(df_s2_diff, on='customer_ID')

    # # after pay (last feature 기준)
    # df_after_day = add_after_pay_features(df)
    # df_after_day = add_period_aggregation(df_after_day, ['last'], df_after_day.columns, [1])
    # df1 = df1.merge(df_after_day, on='customer_ID')

    # lag feature
    num_cols = list(set(df.columns) - set(get_not_used()) - set(get_cat_cols()))
    df_lag = add_lag_feature(df, num_cols, 'mean', 3, 3, '3-3')
    df1 = df1.merge(df_lag, on='customer_ID')
    return df1

def make_preprocessing_2(_df):
    df = _df.copy()
    # columns
    num_cols = list(set(df.columns) - set(get_not_used()) - set(get_cat_cols()))
    cat_cols = get_cat_cols()
    # add sequence
    if 'sequence' not in _df.columns:
        _df = add_sequence(_df)
        df = _df.copy()

    # num agg
    df_stat = add_period_aggregation(df, ['mean', 'last'], num_cols, [13])
    df1 = df_stat

    # cat agg
    df_stat_cat = add_period_aggregation(df, ['last'], cat_cols, [13])
    df1 = df1.merge(df_stat_cat, on='customer_ID')

    # label encoding on cat features (overwrite)
    cat_cols = get_cat_cols_from_list(df1.columns.tolist())
    df1 = encode_cat_label(df1, cat_cols)

    return df1

def make_preprocessing_2(_df):
    df = _df.copy()
    # columns
    num_cols = list(set(df.columns) - set(get_not_used()) - set(get_cat_cols()))
    cat_cols = get_cat_cols()
    # add sequence
    if 'sequence' not in _df.columns:
        _df = add_sequence(_df)
        df = _df.copy()

    # num agg
    df_stat = add_period_aggregation(df, ['mean','last'], num_cols, [13])
    df1 = df_stat

    # cat agg
    df_stat_cat = add_period_aggregation(df, ['last'], cat_cols, [13])
    df1 = df1.merge(df_stat_cat, on='customer_ID')

    # label encoding on cat features (overwrite)
    cat_cols = get_cat_cols_from_list(df1.columns.tolist())
    df1 = encode_cat_label(df1, cat_cols)

    return df1

def split_train_test(df_x, df_y, df_y_test, need_valid_set=False):
    if len(df_x) != (len(df_y) + len(df_y_test)):
        raise ValueError("Arrays must have the same size")
    df_x_train = df_x[df_x['customer_ID'].isin(df_y['customer_ID'])].reset_index(drop=True)
    df_x_train = df_x_train.merge(df_y, on='customer_ID').reset_index(drop=True)
    if need_valid_set:
        df_x_valid = df_x_train.sample(frac=0.1)
        df_x_train = df_x_train.loc[~df_x_train.index.isin(df_x_valid.index)].reset_index(drop=True)
        df_x_valid = df_x_valid.sample(frac=1).reset_index(drop=True)

    df_x_test = df_x[df_x['customer_ID'].isin(df_y_test['customer_ID'])].reset_index(drop=True)
    df_x_test = df_x_test.merge(df_y_test, on='customer_ID')

    df_x_train = df_x_train.sample(frac=1).reset_index(drop=True)
    df_x_test = df_x_test.sample(frac=1).reset_index(drop=True)

    if need_valid_set:
        return df_x_train, df_x_valid, df_x_test
    else:
        return df_x_train, df_x_test

# statement에 날짜별로 sequence 부여 (1: 가장 최근 statement)
def add_sequence(df_x):
    _df_x = df_x.copy()
    _df_x['S_2'] = pd.to_datetime(_df_x['S_2'])
    _df_x['date'] = _df_x['S_2'].dt.year.map(str) + "-" + _df_x['S_2'].dt.month.map("{:02}".format)
    date_list = list(reversed(sorted(_df_x['date'].value_counts().index.tolist())))
    date_dict = {_date: idx + 1 for idx, _date in enumerate(date_list)}
    _df_x['sequence'] = _df_x['date'].replace(date_dict)
    return _df_x

# windows_tail: 최근 N windows만큼 산정
# windows_head: 최고 N windows만큼 산정 (오래된 순)
def add_lag_feature(_df, cols, stat, windows_tail=3, windows_head=3, tag='3-11'):
    df = _df.copy()
    target_col_list = []
    df_1 = add_period_aggregation(df.groupby('customer_ID').tail(windows_tail).reset_index(drop=True), [stat], cols,
                                  [13])
    df_2 = add_period_aggregation(df.groupby('customer_ID').head(windows_head).reset_index(drop=True), [stat], cols,
                                  [13])
    df = df_1.merge(df_2, on='customer_ID')
    for col in cols:
        col1 = None
        col2 = None
        for c in df_1.columns:
            if col in c:
                col1 = c
                break
        for c in df_2.columns:
            if col in c:
                col2 = c
                break
        if (col1 != None) & (col2 != None):
            sub_lag_name = f'_{stat}_{tag}_lag_sub'
            div_lag_name = f'_{stat}_{tag}_lag_div'
            df[col + sub_lag_name] = df_1[col1] - df_2[col2]
            df[col + div_lag_name] = df_1[col1] / df_2[col2]
            target_col_list.append(col)
    return df[['customer_ID'] + [col + sub_lag_name for col in target_col_list] + [col + div_lag_name for col in
                                                                                   target_col_list]]

# compute "after pay" features
def add_after_pay_features(df_x):
    _df_x = df_x.copy()
    feat_list = []
    for bcol in [f'B_{i}' for i in [11, 14, 17]] + ['D_39', 'D_131'] + [f'S_{i}' for i in [16, 23]]:
        for pcol in ['P_2', 'P_3']:
            if bcol in df_x.columns:
                _df_x[f'{bcol}-{pcol}'] = _df_x[bcol] - _df_x[pcol]
                feat_list.append(f'{bcol}-{pcol}')
    return _df_x[['customer_ID', 'S_2'] + feat_list]

# quantile별로 등급 산정
def make_p2_grade_guideline(_df):
    q_list = [0.04, 0.11, 0.23, 0.40, 0.60, 0.77, 0.89, 0.96, 1]
    q_value_list = []
    df = _df.copy()
    for q in q_list:
        q_value_list.append(df['P_2'].quantile(q=q))
    return q_value_list

# P_2 feature 등급 산정
def add_p2_grade(_df):
    df = _df.copy()
    if 'sequence' not in df:
        df = add_sequence(df)
    q_value_list = make_p2_grade_guideline(df.loc[df['sequence'] == 1])
    df['P_2_grade'] = 999
    for idx, q in enumerate(q_value_list):
        df.loc[((df['P_2'] <= q) & (df['P_2_grade'] == 999)), 'P_2_grade'] = len(q_value_list) - idx
    df.loc[df['P_2_grade'] == 999, 'P_2_grade'] = 1
    return df

# 날짜 차이값
def add_s2_diff(_df):
    df = _df.copy()
    df['S_2_diff'] = df[['S_2', 'customer_ID']].groupby('customer_ID').S_2.diff().dt.days
    df_1 = add_period_aggregation(df, ['mean'], ['S_2_diff'], [13])
    df_2 = add_period_aggregation(df, ['std'], ['S_2_diff'], [13])

    return df_1.merge(df_2, on='customer_ID')

# categorical variable > Label Encoding
def encode_cat_label(_df, cat_cols):
    df = _df.copy()
    for col in cat_cols:
        encoder = LabelEncoder()
        df[col] = df[col].astype(str)
        df[col] = encoder.fit_transform(df[col])
    return df
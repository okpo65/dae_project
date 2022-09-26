DAE_CFG = dict(
    hidden_size = 200,
    noise_ratio = .25,
    batch_size = 512,
    init_lr = 3e-5,
    bottleneck_size=50,
    lr_gamma = .995,
    denoise_emphesis = 1
)
BOTTOLENECK_DAE_CFG = dict(
    hidden_size = 50,
    noise_ratio = .3,
    batch_size = 512,
    init_lr = 3e-4,
    bottleneck_size=10,
    lr_gamma = .995,
    denoise_emphesis = .8
)
MLP_CFG = dict(
    hidden_size = 64,
    batch_size = 512,
    init_lr = 5e-5,
    lr_gamma = .995,
    input_dropout = 0.1,
    dropout = .2,
    l2_reg = 2e-3
)
TRANSFORMER_DAE_CFG = dict(
    hidden_size=1024,
    num_subspaces=8,
    embed_dim=128,
    num_heads=8,
    dropout=0.2,
    feedforward_dim=512,
    emphasis=0.8,
    mask_loss_weight=1
)
TRANSFORMER_DAE_TRAIN_CFG = dict(
    init_lr = 1e-4,
    lr_decay = .998,
    max_epochs = 2001
)
lgb_params = {
    'num_iterations': 10500,
    'early_stopping_round': 100,
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting': 'dart',
    'seed': 42,
    'num_leaves': 150,
    'max_depth': 30,
    'learning_rate': 0.001,
    'feature_fraction': 0.60,
    'bagging_freq': 10,
    'bagging_fraction': 0.50,
    'n_jobs': 32,
    'lambda_l2': 2,
    'min_data_in_leaf': 40,
}

WANDB_KEY = '196381c208adc1e785f267c966fe745bf68e987a'
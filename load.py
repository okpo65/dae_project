
import torch
from DAE import DeepStackDAE,DeepBottleneck, TransformerAutoEncoder
from MLP import MLPModel
from config import DAE_CFG, TRANSFORMER_DAE_CFG, MLP_CFG

def load_deepstack_dae_model(len_cat, len_num, model_path=None):
    dae = DeepStackDAE(
        hidden_size=DAE_CFG['hidden_size'],
        num_cats=len_cat,
        num_conts=len_num,
        emphasis=DAE_CFG['denoise_emphesis']
    ).cuda()
    if model_path is not None:
        dae.load_state_dict(torch.load(model_path)['model'])
    return dae

def load_bottleneck_dae_model(len_cat, len_num, model_path=None):
    dae = DeepBottleneck(
        hidden_size=DAE_CFG['hidden_size'],
        bottleneck_size=DAE_CFG['bottleneck_size'],
        num_cats=len_cat,
        num_conts=len_num,
        emphasis=DAE_CFG['denoise_emphesis']
    ).cuda()
    if model_path is not None:
        dae.load_state_dict(torch.load(model_path)['model'])
    return dae

def load_transformer_dae_model(len_cat, len_num, model_path=None):
    dae = TransformerAutoEncoder(
        num_inputs=len_cat + len_num,
        n_cats=len_cat,
        n_nums=len_num,
        **TRANSFORMER_DAE_CFG
    ).cuda()
    if model_path is not None:
        dae.load_state_dict(torch.load(model_path)['model'])
    return dae

def load_mlp_model(model_path, dae_hidden_size, num_hidden_size):
    mlp = MLPModel(
        50,# num_hidden_size * dae_hidden_size,
        hidden_size=MLP_CFG['hidden_size'],
        input_dropout=MLP_CFG['input_dropout'],
        dropout_rate=MLP_CFG['dropout']
    ).cuda()
    mlp.load_state_dict(torch.load(model_path)['model'])
    return mlp
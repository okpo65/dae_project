import torch
import numpy as np
from loss import FocalLoss

bce_logits = torch.nn.functional.binary_cross_entropy_with_logits
mse = torch.nn.functional.mse_loss

class DeepStackDAE(torch.nn.Module):
    def __init__(self,
                 hidden_size=1500,
                 num_cats=10,
                 num_conts=14,
                 emphasis=1,
                 lower_bound=0,
                 upper_bound=10.5):
        super().__init__()
        self.num_cats = num_cats
        self.num_conts = num_conts

        post_encoding_input_size = num_cats + num_conts
        self.linear_1 = torch.nn.Linear(in_features=post_encoding_input_size, out_features=hidden_size)
        self.linear_2 = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.linear_3 = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.linear_4 = torch.nn.Linear(in_features=hidden_size, out_features=post_encoding_input_size)

        self.emphasis = emphasis
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

    def forward(self, x):
        act_1 = torch.nn.functional.relu(self.linear_1(x))
        act_2 = torch.nn.functional.relu(self.linear_2(act_1))
        act_3 = torch.nn.functional.relu(self.linear_3(act_2))
        out = self.linear_4(act_3)
        return act_1, act_2, act_3, out

    def feature(self, x):
        return torch.cat(self.forward(x)[:-1], dim=1)

    def split(self, t):
        return torch.split(t, [self.num_cats, self.num_conts], dim=1)

    def loss(self, x, y, mask=None, weights=[1, 1]):
        if mask is None:
            mask = torch.ones(x.shape).to(x.device)
        loss_weights = mask * self.emphasis + (1 - mask) * (1 - self.emphasis)

        x_cats, x_nums = self.split(self.forward(x)[-1])
        y_cats, y_nums = self.split(y)
        w_cats, w_nums = self.split(mask * self.emphasis + (1 - mask) * (1 - self.emphasis))

        # focal_loss = FocalLoss(size_average=False)
        cat_loss = weights[0] * torch.mul(w_cats, bce_logits(x_cats, y_cats, reduction='none'))
        num_loss = weights[1] * torch.mul(w_nums, mse(x_nums, y_nums, reduction='none'))
        reconstruction_loss = cat_loss.mean() + num_loss.mean()
        return reconstruction_loss

class DeepBottleneck(torch.nn.Module):
    def __init__(self,
                 hidden_size,
                 bottleneck_size,
                 num_cats,
                 num_conts,
                 emphasis=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_cats = num_cats
        self.num_conts = num_conts
        self.emphasis = emphasis
        self.bottleneck_size = bottleneck_size

        post_encoding_input_size = num_cats + num_conts
        half_hidden_size = int(hidden_size / 2)

        print('half_hidden_size!!', half_hidden_size, hidden_size)

        self.batch_norm_1 = torch.nn.BatchNorm1d(hidden_size)
        self.batch_norm_2 = torch.nn.BatchNorm1d(half_hidden_size)
        self.dropout = torch.nn.Dropout(0.2)

        self.encoder_1 = torch.nn.Sequential(
            torch.nn.Linear(in_features=post_encoding_input_size, out_features=hidden_size),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(hidden_size)
        )
        self.encoder_2 = torch.nn.Sequential(
            torch.nn.Linear(in_features=hidden_size, out_features=half_hidden_size),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(half_hidden_size)
        )
        # self.encoder_3 = torch.nn.Sequential(
        #     torch.nn.Linear(in_features=hidden_size, out_features=hidden_size / 2),
        #     torch.nn.ReLU()
        # )
        self.bottleneck_size = torch.nn.Linear(in_features=half_hidden_size, out_features=bottleneck_size)
        self.decoder_1 = torch.nn.Sequential(
            torch.nn.Linear(in_features=bottleneck_size, out_features=half_hidden_size),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(half_hidden_size)
        )
        self.decoder_2 = torch.nn.Sequential(
            torch.nn.Linear(in_features=half_hidden_size, out_features=hidden_size),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(hidden_size)
        )
        self.decoder_3 = torch.nn.Sequential(
            torch.nn.Linear(in_features=hidden_size, out_features=post_encoding_input_size),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(post_encoding_input_size)
        )



    def forward_pass(self, x):
        x = self.encoder_1(x)
        x = self.dropout(x)
        x = self.encoder_2(x)
        x = self.dropout(x)
        x = b = self.bottleneck_size(x)
        x = self.decoder_1(x)
        x = self.dropout(x)
        x = self.decoder_2(x)
        x = self.dropout(x)
        x = self.decoder_3(x)
        return [b, x]
    def forward(self, x):
        return self.forward_pass(x)[1]

    def feature(self, x):
        return self.forward_pass(x)[0]

    def split(self, t):
        return torch.split(t, [self.num_cats, self.num_conts], dim=1)

    def loss(self, x, y, mask=None, weights=[3, 14]):
        if mask is None:
            mask = torch.ones(x.shape).to(x.device)

        x_cats, x_nums = self.split(self.forward(x))
        y_cats, y_nums = self.split(y)
        w_cats, w_nums = self.split(mask * self.emphasis + (1 - mask) * (1 - self.emphasis))

        # focal_loss = FocalLoss(size_average=False)
        cat_loss = 0#weights[0] * torch.mul(w_cats, bce_logits(x_cats, y_cats, reduction='none'))
        num_loss = weights[1] * torch.mul(w_nums, mse(x_nums, y_nums, reduction='none'))
        reconstruction_loss = num_loss.mean()#cat_loss.mean() + num_loss.mean()
        return reconstruction_loss

class TransformerEncoder(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, feedforward_dim):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.linear_1 = torch.nn.Linear(embed_dim, feedforward_dim)
        self.linear_2 = torch.nn.Linear(feedforward_dim, embed_dim)
        self.layernorm_1 = torch.nn.LayerNorm(embed_dim)
        self.layernorm_2 = torch.nn.LayerNorm(embed_dim)

    def forward(self, x_in):
        attn_out, _ = self.attn(x_in, x_in, x_in)
        x = self.layernorm_1(x_in + attn_out)
        ff_out = self.linear_2(torch.nn.functional.relu(self.linear_1(x)))
        x = self.layernorm_2(x + ff_out)
        return x

class TransformerAutoEncoder(torch.nn.Module):
    def __init__(self,
                 num_inputs,
                 n_cats,
                 n_nums,
                 hidden_size=1024,
                 num_subspaces=8,
                 embed_dim=128,
                 num_heads=8,
                 dropout=0,
                 feedforward_dim=512,
                 emphasis=0.75,
                 task_weights=[10,25],
                 mask_loss_weight=2):
        super().__init__()
        assert hidden_size == embed_dim * num_subspaces
        self.n_cats = n_cats
        self.n_nums = n_nums
        self.num_subspaces = num_subspaces
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.emphasis = emphasis
        self.task_weights = np.array(task_weights) / sum(task_weights)
        self.mask_loss_weight = mask_loss_weight

        self.excite = torch.nn.Linear(in_features=num_inputs, out_features=hidden_size)
        self.encoder_1 = TransformerEncoder(embed_dim, num_heads, dropout, feedforward_dim)
        self.encoder_2 = TransformerEncoder(embed_dim, num_heads, dropout, feedforward_dim)
        self.encoder_3 = TransformerEncoder(embed_dim, num_heads, dropout, feedforward_dim)

        self.mask_predictor = torch.nn.Linear(in_features=hidden_size, out_features=num_inputs)
        self.reconstructor = torch.nn.Linear(in_features=hidden_size+num_inputs, out_features=num_inputs)

    def divide(self, x):
        batch_size = x.shape[0]
        x = x.reshape((batch_size, self.num_subspaces, self.embed_dim)).permute((1, 0, 2))
        return x

    def combine(self, x):
        batch_size = x.shape[1]
        x = x.permute((1, 0, 2)).reshape((batch_size, -1))
        return x

    def forward(self, x):
        x = torch.nn.functional.relu(self.excite(x))

        x = self.divide(x)
        x1 = self.encoder_1(x)
        x2 = self.encoder_2(x1)
        x3 = self.encoder_3(x2)
        x = self.combine(x3)

        predicted_mask = self.mask_predictor(x)
        reconstruction = self.reconstructor(torch.cat([x, predicted_mask], dim=1))
        return (x1, x2, x3), (reconstruction, predicted_mask)

    def split(self, t):
        return torch.split(t, [self.n_cats, self.n_nums], dim=1)

    def feature(self, x):
        attn_outs, _ = self.forward(x)
        return torch.cat([self.combine(x) for x in attn_outs], dim=1)

    def loss(self, x, y, mask, reduction='mean'):
        _, (reconstruction, predicted_mask) = self.forward(x)
        x_cats, x_nums = self.split(reconstruction)
        y_cats, y_nums = self.split(y)
        w_cats, w_nums = self.split(mask * self.emphasis + (1 - mask) * (1 - self.emphasis))

        cat_loss = self.task_weights[0] * torch.mul(w_cats, bce_logits(x_cats, y_cats, reduction='none'))
        num_loss = self.task_weights[1] * torch.mul(w_nums, mse(x_nums, y_nums, reduction='none'))

        reconstruction_loss = torch.cat([cat_loss, num_loss],
                                        dim=1) if reduction == 'none' else cat_loss.mean() + num_loss.mean()
        mask_loss = self.mask_loss_weight * bce_logits(predicted_mask, mask, reduction=reduction)

        return reconstruction_loss + mask_loss if reduction == 'mean' else [reconstruction_loss, mask_loss]
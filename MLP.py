import torch
class MLPModel(torch.nn.Module):
    def __init__(self,
                 num_inputs,
                 hidden_size=512,
                 input_dropout=0,
                 dropout_rate=0.2,
                 lower_bound=0,
                 upper_bound=10.5):
        super().__init__()
        self.use_input_dropout = input_dropout > 0

        half_hidden_size = int(hidden_size / 2)

        if input_dropout:
            self.input_dropout = torch.nn.Dropout(input_dropout)

        self.layer_1 = torch.nn.Sequential(
            torch.nn.Linear(num_inputs, hidden_size),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(hidden_size),
            torch.nn.Dropout(dropout_rate)
        )
        self.layer_2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(hidden_size),
            torch.nn.Dropout(dropout_rate)
        )

        self.layer_3 = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, half_hidden_size),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(half_hidden_size),
            torch.nn.Dropout(dropout_rate)
        )

        self.layer_4 = torch.nn.Sequential(
            torch.nn.Linear(half_hidden_size, half_hidden_size),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(half_hidden_size),
            torch.nn.Dropout(dropout_rate)
        )
        self.dropout = torch.nn.Dropout(dropout_rate)

        self.last_linear = torch.nn.Linear(half_hidden_size, 1)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def net(self, x):
        if self.use_input_dropout:
            x = self.input_dropout(x)
        x = self.layer_1(x)
        # x = self.layer_2(x)
        x = self.layer_3(x)
        # x = self.layer_4(x)
        return self.last_linear(x)

    def forward(self, x):
        return torch.sigmoid(self.net(x) * (self.upper_bound - self.lower_bound) + self.lower_bound)
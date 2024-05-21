import torch

class MultiHeadAttention(torch.nn.Module):

    def __init__(self,
                 d_model: int,
                 q: int,
                 v: int,
                 h: int):
        super(MultiHeadAttention, self).__init__()

        self.h = h
        self.q = q

        self.W_Q = torch.nn.Linear(in_features=d_model, out_features=q * h)
        self.W_K = torch.nn.Linear(in_features=d_model, out_features=q * h)
        self.W_V = torch.nn.Linear(in_features=d_model, out_features=v * h)
        self.W_out = torch.nn.Linear(in_features=v * h, out_features=d_model)

        self.inf = -2**32+1
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    def forward(self,
                x: torch.Tensor,
                stage: str):

        Q = torch.cat(self.W_Q(x).chunk(self.h, dim=-1), dim=0)
        K = torch.cat(self.W_K(x).chunk(self.h, dim=-1), dim=0)
        V = torch.cat(self.W_V(x).chunk(self.h, dim=-1), dim=0)

        score = torch.matmul(Q, K.transpose(-1, -2))  # / torch.sqrt(torch.Tensor(self.q)).to(self.device)

        heatmap_score = score

        if stage == 'train':
            mask = torch.ones_like(score[0])
            mask = mask.tril(diagonal=0)
            score = torch.where(mask > 0, score, (torch.ones_like(mask) * self.inf).to(self.device))

        score = torch.nn.functional.softmax(score, dim=-1)
        weight_V = torch.cat(torch.matmul(score, V).chunk(self.h, dim=0), dim=-1)

        out = self.W_out(weight_V)

        return out, heatmap_score
    
class PositionFeedforward(torch.nn.Module):
    def __init__(self,
                 d_model: int,
                 d_hidden: int = 2048):
        super(PositionFeedforward, self).__init__()

        self.linear1 = torch.nn.Linear(in_features=d_model, out_features=d_hidden)
        self.linear2 = torch.nn.Linear(in_features=d_hidden, out_features=d_model)
        self.relu = torch.nn.ReLU()
        self.layernorm = torch.nn.LayerNorm(normalized_shape=d_model)

    def forward(self, x):

        residual = x
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.layernorm(x + residual)

        return x
    
class Encoder(torch.nn.Module):
    def __init__(self,
                 q: int,
                 v: int,
                 h: int,
                 d_model: int,
                 d_hidden: int,
                 dropout: float = 0.2):
        super(Encoder, self).__init__()

        self.MHA = MultiHeadAttention(d_model=d_model, q=q, v=v, h=h)
        self.feedforward = PositionFeedforward(d_model=d_model, d_hidden=d_hidden)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.layernorm = torch.nn.LayerNorm(d_model)

    def forward(self,
                x: torch.Tensor,
                stage: str):
        residual = x
        x, heatmap_score = self.MHA(x, stage)
        x = self.dropout(x)
        x = self.layernorm(x + residual)

        x = self.feedforward(x)

        return x, heatmap_score
    

class Embedding(torch.nn.Module):
    def __init__(self,
                 d_feature: int,
                 d_timestep: int,
                 d_model: int,
                 wise: str = 'timestep' or 'feature'):
        super(Embedding, self).__init__()

        assert wise == 'timestep' or wise == 'feature', 'Embedding wise error!'
        self.wise = wise

        if wise == 'timestep':
            self.embedding = torch.nn.Linear(d_feature, d_model)
        elif wise == 'feature':
            self.embedding = torch.nn.Linear(d_timestep, d_model)

    def forward(self,
                x: torch.Tensor):
        if self.wise == 'feature':
            x = self.embedding(x)
        elif self.wise == 'timestep':
            x = self.embedding(x.transpose(-1, -2))
            x = position_encode(x)

        return x, None


def position_encode(x):

    pe = torch.ones_like(x[0])
    position = torch.arange(0, x.shape[1]).unsqueeze(-1)
    temp = torch.Tensor(range(0, x.shape[-1], 2))
    temp = temp * -(math.log(10000) / x.shape[-1])
    temp = torch.exp(temp).unsqueeze(0)
    temp = torch.matmul(position.float(), temp)  # shape:[input, d_model/2]
    pe[:, 0::2] = torch.sin(temp)
    pe[:, 1::2] = torch.cos(temp)

    return x + pe

class Transformer(torch.nn.Module):
    def __init__(self,
                 d_model: int,
                 d_hidden: int,
                 d_feature: int,
                 d_timestep: int,
                 q: int,
                 v: int,
                 h: int,
                 N: int,
                 class_num: int,
                 dropout: float = 0.2):
        super(Transformer, self).__init__()

        self.timestep_embedding = Embedding(d_feature=d_feature, d_timestep=d_timestep, d_model=d_model, wise='timestep')
        self.feature_embedding = Embedding(d_feature=d_feature, d_timestep=d_timestep, d_model=d_model, wise='feature')

        self.timestep_encoderlist = torch.nn.ModuleList([Encoder(
            d_model=d_model,
            d_hidden=d_hidden,
            q=q,
            v=v,
            h=h,
            dropout=dropout) for _ in range(N)])

        self.feature_encoderlist = torch.nn.ModuleList([Encoder(
            d_model=d_model,
            d_hidden=d_hidden,
            q=q,
            v=v,
            h=h,
            dropout=dropout) for _ in range(N)])

        self.gate = torch.nn.Linear(in_features=d_timestep * d_model + d_feature * d_model, out_features=2)
        self.linear_out = torch.nn.Linear(in_features=d_timestep * d_model + d_feature * d_model,
                                          out_features=class_num)

    def forward(self,
                x: torch.Tensor,
                stage: str = 'train' or 'test'):

        x_timestep, _ = self.timestep_embedding(x)
        x_feature, _ = self.feature_embedding(x)

        for encoder in self.timestep_encoderlist:
            x_timestep, heatmap = encoder(x_timestep, stage=stage)

        for encoder in self.feature_encoderlist:
            x_feature, heatmap = encoder(x_feature, stage=stage)

        x_timestep = x_timestep.reshape(x_timestep.shape[0], -1)
        x_feature = x_feature.reshape(x_feature.shape[0], -1)

        gate = torch.nn.functional.softmax(self.gate(torch.cat([x_timestep, x_feature], dim=-1)), dim=-1)

        gate_out = torch.cat([x_timestep * gate[:, 0:1], x_feature * gate[:, 1:2]], dim=-1)

        out = self.linear_out(gate_out)

        return out

class Global_Dataset(Dataset):
    def __init__(self, X, Y):
        super(Global_Dataset, self).__init__()
        self.X = X
        self.Y = Y
        self.Y[self.Y == -1] = 0
        self.min_label = min(self.Y)
        self.Y = self.Y - self.min_label
        self.data_len = len(self.X)

    def __getitem__(self, item):
        return self.X[item], self.Y[item]

    def __len__(self):
        return self.data_len
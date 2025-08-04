import torch
from torch import nn
import torch.nn.functional as F


class FinalP2TLayer(nn.Module):
    def __init__(
        self,
        in_place_channel,
        num_classes,
        act=F.gelu,
        dropout=0.1,
    ):
        super(FinalP2TLayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_classes = num_classes
        self.fc_p2t = nn.Linear(
            in_features=in_place_channel,
            out_features=num_classes * 2 * int(in_place_channel),
            bias=False
        )
        self.act = act

    def forward(
        self,
        C_t_stack,
        place_features
    ):
        tmp = self.fc_p2t(place_features).view(
            place_features.shape[0],
            place_features.shape[1],
            self.num_classes * 2, -1
        ).permute(0, 2, 1, 3)
        return torch.matmul(C_t_stack, tmp).sum(1)


class PNCLayer(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        num_classes,
        expand_ratio,
        act=F.gelu,
        dropout=0.1,
    ):
        super(PNCLayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_classes = num_classes
        self.fc_p2t = nn.Linear(
            in_features=in_channel,
            out_features=num_classes * 2 * int(in_channel * expand_ratio),
            bias=False
        )
        self.fc_t2p = nn.Linear(
            in_features=int(in_channel * expand_ratio),
            out_features=num_classes * 2 * in_channel,
            bias=False
        )
        self.fc = nn.Linear(in_features=in_channel, out_features=out_channel, bias=True)
        self.act = act

    def forward(
        self,
        C_t_stack,
        C_stack,
        place_features
    ):
        tmp = self.fc_p2t(place_features).view(
            place_features.shape[0],
            place_features.shape[1],
            self.num_classes * 2, -1
        ).permute(0, 2, 1, 3)
        tmp = self.act(torch.matmul(C_t_stack, tmp).sum(1))
        tmp = self.dropout(tmp)

        tmp = self.fc_t2p(tmp).view(
            tmp.shape[0],
            tmp.shape[1],
            self.num_classes * 2, -1
        ).permute(0, 2, 1, 3)
        place_features = self.act(torch.matmul(C_stack, tmp).sum(1) + place_features)
        place_features = self.dropout(place_features)
        return self.fc(place_features)

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertEncoder


class FinalP2TLayer(nn.Module):
    def __init__(
        self,
        in_place_channel,
        num_classes,
    ):
        super(FinalP2TLayer, self).__init__()
        self.num_classes = num_classes
        self.fc_p2t = nn.Linear(
            in_features=in_place_channel,
            out_features=num_classes * 2 * int(in_place_channel),
            bias=False
        )

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
        act=F.leaky_relu
    ):
        super(PNCLayer, self).__init__()
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

        tmp = self.fc_t2p(tmp).view(
            tmp.shape[0],
            tmp.shape[1],
            self.num_classes * 2, -1
        ).permute(0, 2, 1, 3)
        place_features = self.act(torch.matmul(C_stack, tmp).sum(1) + place_features)
        return self.act(self.fc(place_features))


class PNCN(nn.Module):
    def __init__(
        self, 
        num_classes, 
        in_channel, 
        num_pnc_layers, 
        hidden_channel, 
        expand_ratio,
        act=F.leaky_relu,
        num_transformer_layers=3,
        num_attention_heads=4,
        transformer_intermediate_size=768,
    ):
        super(PNCN, self).__init__()
        self.act = act
        self.num_classes = num_classes
        self.num_pnc_layers = num_pnc_layers

        self.fc = torch.nn.Linear(in_channel, hidden_channel)

        self.pnc_layers = nn.ModuleList([PNCLayer(num_classes=self.num_classes, in_channel=hidden_channel,
                                                  out_channel=hidden_channel, expand_ratio=expand_ratio,
                                                  act=act)] +
                                        [PNCLayer(num_classes=self.num_classes, in_channel=hidden_channel,
                                                  out_channel=hidden_channel, expand_ratio=expand_ratio,
                                                  act=act)
                                         for _ in range(num_pnc_layers - 1)])

        self.final_p2t = FinalP2TLayer(num_classes=self.num_classes,
                                       in_place_channel=hidden_channel)

        # 配置模型参数
        config = BertConfig(
            hidden_size=hidden_channel,  # 输入输出的隐藏层维度
            num_hidden_layers=num_transformer_layers,  # 只使用一个Transformer层
            num_attention_heads=num_attention_heads,  # 注意力头数（768/64=12）
            intermediate_size=transformer_intermediate_size,  # FFN中间层维度（默认值）
            position_embedding_type="none",  # 关键：禁用位置编码
        )
        self.final_transformer = BertEncoder(config)

        self.final_fc = nn.Linear(
            in_features=hidden_channel,
            out_features=1,
            bias=True
        )

    def forward(
        self, 
        x, 
        C_t_stack, 
        C_stack,
        attention_mask,  # [B, |T| + |P|]
    ):
        x = self.act(self.fc(x))
        for pnc_layer in self.pnc_layers:
            x = pnc_layer(C_t_stack=C_t_stack, C_stack=C_stack, place_features=x)
        x_t = self.final_p2t(C_t_stack=C_t_stack, place_features=x)
        res = self.final_transformer(
            torch.cat([x_t, x], dim=1),
            attention_mask=attention_mask.unsqueeze(1).unsqueeze(2),
        )['last_hidden_state']
        logits = self.final_fc(res[:, :x_t.shape[1]])
        return logits

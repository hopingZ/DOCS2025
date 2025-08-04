import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertConfig
from models.layers import PNCLayer, FinalP2TLayer
from transformers.models.bert.modeling_bert import BertEncoder


class PNCN(nn.Module):
    def __init__(
        self,
        num_classes,
        in_channel,
        num_pnc_layers,
        hidden_channel,
        expand_ratio,
        act=F.gelu,
        num_transformer_layers=3,
        num_attention_heads=4,
        transformer_intermediate_size=768,
        dropout=0.1,
    ):
        super(PNCN, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.act = act
        self.num_classes = num_classes
        self.num_pnc_layers = num_pnc_layers

        self.fc = torch.nn.Linear(in_channel, hidden_channel)

        self.pnc_layers = nn.ModuleList([PNCLayer(num_classes=self.num_classes, in_channel=hidden_channel,
                                                  out_channel=hidden_channel, expand_ratio=expand_ratio,
                                                  act=act, dropout=dropout)] +
                                        [PNCLayer(num_classes=self.num_classes, in_channel=hidden_channel,
                                                  out_channel=hidden_channel, expand_ratio=expand_ratio,
                                                  act=act, dropout=dropout)
                                         for _ in range(num_pnc_layers - 1)])

        self.final_p2t = FinalP2TLayer(
            num_classes=self.num_classes,
            in_place_channel=hidden_channel,
            act=act,
            dropout=dropout,
        )

        # 配置模型参数
        config = BertConfig(
            hidden_size=hidden_channel,  # 输入输出的隐藏层维度
            num_hidden_layers=num_transformer_layers,  # 层数
            num_attention_heads=num_attention_heads,  # 注意力头数
            intermediate_size=transformer_intermediate_size,  # FFN中间层维度（默认值）
            position_embedding_type="none",  # 关键：禁用位置编码
            hidden_dropout_prob=dropout,
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
        x = self.fc(x)

        for pnc_layer in self.pnc_layers:
            x = F.layer_norm(x, normalized_shape=x.shape)
            x = self.act(x)
            x = self.dropout(x)
            x = pnc_layer(C_t_stack=C_t_stack, C_stack=C_stack, place_features=x)

        x_t = self.final_p2t(
            C_t_stack=C_t_stack,
            place_features=self.dropout(self.act(F.layer_norm(x, normalized_shape=x.shape))),
        )

        res = self.final_transformer(
            torch.cat([x_t, x], dim=1),
            attention_mask=attention_mask.unsqueeze(1).unsqueeze(2),
        )['last_hidden_state']
        logits = self.final_fc(res[:, :x_t.shape[1]])
        logits = -torch.nn.functional.leaky_relu(logits)
        return logits

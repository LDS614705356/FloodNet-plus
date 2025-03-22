import torch
from torch import nn
from torch.nn import functional as F
from models.common.attention import MultiHeadAttention
from models.common.pos_embed import sinusoid_encoding_table, PositionWiseFeedForward
from einops import repeat

########
#mplug
########

class TransformerLayer(nn.Module):

    def __init__(
            self,
            d_model=512,
            n_heads=8,
            d_ff=2048,
            dropout=.1,
            attn_dropout=0.0,
            attention_module=None,
            **kwargs,
    ):
        super(TransformerLayer, self).__init__()
        self.mhatt = MultiHeadAttention(
            d_model,
            n_heads,
            dropout,
            attention_module=attention_module,
            attn_dropout=attn_dropout,
            **kwargs,
        )
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        att = self.mhatt(queries, keys, values, attention_mask, attention_weights)
        ff = self.pwff(att)
        return ff

class GateTransformerLayer(nn.Module):

    def __init__(
            self,
            d_model=512,
            n_heads=8,
            d_ff=2048,
            dropout=.1,
            attn_dropout=0.0,
            attention_module=None,
            **kwargs,
    ):
        super(GateTransformerLayer, self).__init__()
        self.mhatt = MultiHeadAttention(
            d_model,
            n_heads,
            dropout,
            attention_module=attention_module,
            attn_dropout=attn_dropout,
            **kwargs,
        )
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        att = self.mhatt(queries, keys, values, attention_mask, attention_weights)
        ff = self.pwff(att)
        gating = self.linear(att)
        gating = torch.sigmoid(gating)
        return ff, gating

class GuideTransformerLayer(nn.Module):

    def __init__(
            self,
            d_model=512,
            n_heads=8,
            d_ff=2048,
            dropout=.1,
            attn_dropout=0.0,
            attention_module=None,
            **kwargs,
    ):
        super(GuideTransformerLayer, self).__init__()
        self.mhatt = MultiHeadAttention(
            d_model,
            n_heads,
            dropout,
            attention_module=attention_module,
            attn_dropout=attn_dropout,
            **kwargs,
        )
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

    def forward(self, queries, keys, values, gating, attention_mask=None, attention_weights=None):
        att = self.mhatt(queries, keys, values, attention_mask, attention_weights)
        att = att * gating
        ff = self.pwff(att)
        return ff


class GridFeatureNetwork(nn.Module):

    def __init__(
            self,
            vocab_size,
            max_len,
            n_layers,
            pad_idx=1,
            d_in=1024,
            d_model=512,
            n_heads=8,
            d_ff=2048,
            dropout=0.1,
            attn_dropout=0.0,
            attention_module=None,
            **kwargs,
    ):
        super().__init__()
        self.fc = nn.Linear(d_in, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self._is_stateful = False
        self.pad_idx = 1
        self.word_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len + 1, d_model, 0), freeze=True)

        self.layers = nn.ModuleList([
            TransformerLayer(
                d_model,
                n_heads,
                d_ff,
                dropout,
                attn_dropout=attn_dropout,
                attention_module=attention_module,
                **kwargs,
            ) for _ in range(n_layers)
        ])

        for layer in self.layers:
            for name, value in layer.named_parameters():
                value.requires_grad = False
        for name, value in self.fc.named_parameters():
            value.requires_grad = False
        for name, value in self.dropout.named_parameters():
            value.requires_grad = False
        for name, value in self.layer_norm.named_parameters():
            value.requires_grad = False

        self.text_layers = nn.ModuleList([
            GateTransformerLayer(
                d_model,
                n_heads,
                d_ff,
                dropout,
                attn_dropout=attn_dropout,
                attention_module=attention_module,
                **kwargs,
            ) for _ in range(n_layers)
        ])

        self.co_layer = nn.ModuleList([
            GuideTransformerLayer(
                d_model,
                n_heads,
                d_ff,
                dropout,
                attn_dropout=attn_dropout,
                attention_module=attention_module,
                **kwargs,
            ) for _ in range(n_layers)
        ])

        self.concat_layer = TransformerLayer(
            d_model,
            n_heads,
            d_ff,
            dropout,
            attn_dropout=attn_dropout,
            attention_module=attention_module,
            **kwargs,
        )
        # self.fc2 = nn.Linear(d_in, d_model)

    def get_seq_inputs(self, input):
        # input (b_s, seq_len); when use beam search: input [BB 1]
        b_s, seq_len = input.shape[:2]
        mask_pad = (input != self.pad_idx).unsqueeze(-1).float()  # (b_s, seq_len, 1)
        mask_x = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8, device=input.device), diagonal=1)  # 上三角
        mask_x = mask_x.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        mask_x = mask_x + (input == self.pad_idx).unsqueeze(1).unsqueeze(1).byte()
        mask_x = mask_x.gt(0)  # (b_s, 1, seq_len, seq_len)
        if self._is_stateful:
            self.running_mask_x = torch.cat([self.running_mask_x, mask_x], -1)
            mask_x = self.running_mask_x

        seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(input.device)  # (b_s, seq_len)
        seq = seq.masked_fill(mask_pad.squeeze(-1) == 0, 0)
        if self._is_stateful:
            self.running_seq.add_(1)
            seq = self.running_seq
        x = self.word_emb(input) + self.pos_emb(seq)

        return {
            'mask_pad': mask_pad,  #
            'mask_x': mask_x,
            'seq': seq,
            'x': x,
        }

    def forward(self, input, seq_input, attention_mask=None, attention_weights=None):
        seq_inputs = self.get_seq_inputs(seq_input)
        mask_pad = seq_inputs['mask_pad']
        mask_y = seq_inputs['mask_x']
        y = seq_inputs['x']

        out = F.relu(self.fc(input))
        out = self.dropout(out)
        out = self.layer_norm(out)

        # if attention_mask is None:
        #     attention_mask = (torch.sum(out, dim=-1) == self.padding_idx)
        #     attention_mask = repeat(attention_mask, 'B N -> B 1 1 N')  # [B Head Nq N]

        outs = []
        text_outs = []
        for l, tl, cl in zip(self.layers, self.text_layers, self.co_layer):
            out = l(out, out, out, attention_mask, attention_weights)
            y, gating = tl(y, y, y, None, None)
            y = cl(y, out, out, gating, None, None)
            outs.append(out.unsqueeze(1))
            text_outs.append(y.unsqueeze(1))

        outs_concat = torch.cat([outs[-1], text_outs[-1]], dim=2)
        outs_concat = outs_concat.squeeze(1)
        outs_concat = self.concat_layer(outs_concat,outs_concat,outs_concat,None,None)
        # outs_concat = self.fc2(outs_concat)
        outs[-1] = outs_concat.unsqueeze(1)

        # outs = torch.cat(outs, 1)
        return outs_concat, attention_mask

import math

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, repeat


class junction(nn.Module):
    def __init__(self, num_patches, patch_dim, dim, patch_h, patch_w): #N, (CHW), output
        super(junction, self).__init__()
        self.num_patches = num_patches
        self.dim = dim
        # self.fc_spatial = nn.Linear(patch_dim_spatial, dim)
        # self.fc_channel = nn.Linear(patch_dim_channel, dim)
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim))
        self.dropout = nn.Dropout(0.1)
        self.avgpool1d = torch.nn.AdaptiveAvgPool1d(1)
        self.conv_spatial = nn.ModuleList([
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5, padding=2)
            for _ in range(3)
        ])
        # self.conv_channel = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5, padding=2)
        self.sigmoid = nn.Sigmoid()
        # nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.fc_s = nn.ModuleList([
            nn.Linear(patch_h*patch_w, 2)
            for _ in range(512)
        ])
        self.fc = nn.Linear(dim, dim)

        for name, value in self.fc.named_parameters():
            value.requires_grad = False
        # self.pos_embed.requires_grad = False
        for name, value in self.dropout.named_parameters():
            value.requires_grad = False

        for name, value in self.avgpool1d.named_parameters():
            value.requires_grad = False
        for l in self.conv_spatial:
            for name, value in l.named_parameters():
                value.requires_grad = False

        # self.conv = nn.Conv2d(patch_dim, dim, (patch_h, patch_w))
        self._init_params()


    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)
        pass

    def forward(self,x):  # x: [b, 100(N), C(512), H(19), W(25)]
        torch.cuda.empty_cache()

        for i in range(len(self.conv_spatial)):
            x_spatial = rearrange(x, 'B N C H W -> B N (C H W)')
            # x_channel = rearrange(x, 'B N C H W -> B C (N H W)')
            x_spatial_att = self.avgpool1d(x_spatial)
            # x_channel_att = self.avgpool1d(x_channel)

            x_spatial_att = rearrange(x_spatial_att, 'B N n -> B n N')
            # x_channel_att = rearrange(x_channel_att, 'B C n -> B n C')

            x_spatial_att = self.conv_spatial[i](x_spatial_att)
            x_spatial_att = self.sigmoid(x_spatial_att)
            # x_channel_att = self.conv_channel(x_channel_att)
            # x_channel_att = self.sigmoid(x_channel_att)

            # x_channel_att = rearrange(x_channel_att, 'B n C -> B C n')
            # x_channel_att.squeeze(-1)
            # x_channel_att = repeat(x_channel_att, 'B C -> B C N H W', N=x.size(1), H=x.size(-2), W=x.size(-1))
            x_spatial_att = rearrange(x_spatial_att, 'B n N -> B N n')
            x_spatial_att = x_spatial_att.squeeze(-1)
            x_spatial_att = repeat(x_spatial_att, 'B N -> B N C H W', C=x.size(-3), H=x.size(-2), W=x.size(-1))
            # x_att = x_spatial_att * x_channel_att
            # x_att = self.sigmoid(x_att)
            # x_att = repeat(x_att, 'B N C -> B N C H W', H=x.size(-2), W=x.size(-1))
            x = x * x_spatial_att

        x = rearrange(x, 'B N C H W -> B N C (H W)')
        x_list = torch.split(x, split_size_or_sections=1, dim=2)
        y_list = []
        for x_split, fc_layer in zip(x_list, self.fc_s):
            x_split = fc_layer(x_split)
            y_list.append(x_split)

        x = torch.cat(y_list, dim=2)
        x = rearrange(x, 'B N C D -> B N (C D)')
        x = self.fc(x)
        x = self.dropout(x)

        # x = rearrange(x, 'B N C H W -> (B N) C H W')
        # bn = x.shape[0]
        # x = self.conv(x)
        # x = x.squeeze(3).squeeze(2)
        # x = rearrange(x, '(B N) C -> B N C', N=self.num_patches, B=bn // self.num_patches)
        #
        # x = self.dropout(x + self.pos_embed)
        # x_channel_att = rearrange(x_channel_att, 'B n C -> B C n')

        # x_spatial_att = self.sigmoid(x_spatial_att)
        # x_channel_att = self.sigmoid(x_channel_att)
        #
        # x_spatial = x_spatial * x_spatial_att.expand_as(x_spatial)
        # x_channel = x_channel * x_channel_att.expand_as(x_channel)

        return x                            # x:[B, N, dim]
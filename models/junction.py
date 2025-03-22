import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, repeat


class junction(nn.Module):
    def __init__(self, num_patches, patch_dim, dim): #N, (CHW), output
        super(junction, self).__init__()
        self.fc = nn.Linear(patch_dim, dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim))
        self.dropout = nn.Dropout(0.1)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        for name, value in self.fc.named_parameters():
            value.requires_grad = False
        self.pos_embed.requires_grad = False
        for name, value in self.dropout.named_parameters():
            value.requires_grad = False
        self._init_params()


    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
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

    def forward(self,x):  # x: [b, 100(N), C(16?), H(38?), W(50?)]
        torch.cuda.empty_cache()
        x = rearrange(x, 'B N C H W -> B N (C H W)')
        x = self.fc(x)
        x = self.dropout(x + self.pos_embed)
        return x                            # x:[B, N, dim]
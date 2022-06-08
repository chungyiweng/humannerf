import torch
import torch.nn as nn

from core.utils.network_util import initseq


class CanonicalMLP(nn.Module):
    def __init__(self, mlp_depth=8, mlp_width=256, 
                 input_ch=3, skips=None,
                 **_):
        super(CanonicalMLP, self).__init__()

        if skips is None:
            skips = [4]

        self.mlp_depth = mlp_depth
        self.mlp_width = mlp_width
        self.input_ch = input_ch
        
        pts_block_mlps = [nn.Linear(input_ch, mlp_width), nn.ReLU()]

        layers_to_cat_input = []
        for i in range(mlp_depth-1):
            if i in skips:
                layers_to_cat_input.append(len(pts_block_mlps))
                pts_block_mlps += [nn.Linear(mlp_width + input_ch, mlp_width), 
                                   nn.ReLU()]
            else:
                pts_block_mlps += [nn.Linear(mlp_width, mlp_width), nn.ReLU()]
        self.layers_to_cat_input = layers_to_cat_input

        self.pts_linears = nn.ModuleList(pts_block_mlps)
        initseq(self.pts_linears)

        # output: rgb + sigma (density)
        self.output_linear = nn.Sequential(nn.Linear(mlp_width, 4))
        initseq(self.output_linear)


    def forward(self, pos_embed, **_):
        h = pos_embed
        for i, _ in enumerate(self.pts_linears):
            if i in self.layers_to_cat_input:
                h = torch.cat([pos_embed, h], dim=-1)
            h = self.pts_linears[i](h)

        outputs = self.output_linear(h)

        return outputs    
        
import torch
import torch.nn as nn

from core.utils.network_util import initseq


class NonRigidMotionMLP(nn.Module):
    def __init__(self,
                 pos_embed_size=3, 
                 condition_code_size=69,
                 mlp_width=128,
                 mlp_depth=6,
                 skips=None):
        super(NonRigidMotionMLP, self).__init__()

        self.skips = [4] if skips is None else skips
        
        block_mlps = [nn.Linear(pos_embed_size+condition_code_size, 
                                mlp_width), nn.ReLU()]
        
        layers_to_cat_inputs = []
        for i in range(1, mlp_depth):
            if i in self.skips:
                layers_to_cat_inputs.append(len(block_mlps))
                block_mlps += [nn.Linear(mlp_width+pos_embed_size, mlp_width), 
                               nn.ReLU()]
            else:
                block_mlps += [nn.Linear(mlp_width, mlp_width), nn.ReLU()]

        block_mlps += [nn.Linear(mlp_width, 3)]

        self.block_mlps = nn.ModuleList(block_mlps)
        initseq(self.block_mlps)

        self.layers_to_cat_inputs = layers_to_cat_inputs

        # init the weights of the last layer as very small value
        # -- at the beginning, we hope non-rigid offsets are zeros 
        init_val = 1e-5
        last_layer = self.block_mlps[-1]
        last_layer.weight.data.uniform_(-init_val, init_val)
        last_layer.bias.data.zero_()


    def forward(self, pos_embed, pos_xyz, condition_code, viewdirs=None, **_):
        h = torch.cat([condition_code, pos_embed], dim=-1)
        if viewdirs is not None:
            h = torch.cat([h, viewdirs], dim=-1)

        for i in range(len(self.block_mlps)):
            if i in self.layers_to_cat_inputs:
                h = torch.cat([h, pos_embed], dim=-1)
            h = self.block_mlps[i](h)
        trans = h

        result = {
            'xyz': pos_xyz + trans,
            'offsets': trans
        }
        
        return result

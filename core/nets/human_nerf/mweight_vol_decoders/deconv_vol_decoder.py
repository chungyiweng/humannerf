import torch
import torch.nn as nn
import torch.nn.functional as F

from core.utils.network_util import ConvDecoder3D


class MotionWeightVolumeDecoder(nn.Module):
    def __init__(self, embedding_size=256, volume_size=32, total_bones=24):
        super(MotionWeightVolumeDecoder, self).__init__()

        self.total_bones = total_bones
        self.volume_size = volume_size
        
        self.const_embedding = nn.Parameter(
            torch.randn(embedding_size), requires_grad=True 
        )

        self.decoder = ConvDecoder3D(
            embedding_size=embedding_size,
            volume_size=volume_size, 
            voxel_channels=total_bones+1)


    def forward(self,
                motion_weights_priors,
                **_):
        embedding = self.const_embedding[None, ...]
        decoded_weights =  F.softmax(self.decoder(embedding) + \
                                        torch.log(motion_weights_priors), 
                                     dim=1)
        
        return decoded_weights

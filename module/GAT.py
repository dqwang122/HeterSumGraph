#!/usr/bin/python
# -*- coding: utf-8 -*-

# __author__="Danqing Wang"

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from module.GATStackLayer import MultiHeadSGATLayer, MultiHeadLayer
from module.GATLayer import PositionwiseFeedForward, WSGATLayer, SWGATLayer

######################################### SubModule #########################################
class WSWGAT(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, attn_drop_out, ffn_inner_hidden_size, ffn_drop_out, feat_embed_size, layerType):
        super().__init__()
        self.layerType = layerType
        if layerType == "W2S":
            self.layer = MultiHeadLayer(in_dim, int(out_dim / num_heads), num_heads, attn_drop_out, feat_embed_size, layer=WSGATLayer)
        elif layerType == "S2W":
            self.layer = MultiHeadLayer(in_dim, int(out_dim / num_heads), num_heads, attn_drop_out, feat_embed_size, layer=SWGATLayer)
        elif layerType == "S2S":
            self.layer = MultiHeadSGATLayer(in_dim, int(out_dim / num_heads), num_heads, attn_drop_out)
        else:
            raise NotImplementedError("GAT Layer has not been implemented!")

        self.ffn = PositionwiseFeedForward(out_dim, ffn_inner_hidden_size, ffn_drop_out)

    def forward(self, g, w, s):
        if self.layerType == "W2S":
            origin, neighbor = s, w
        elif self.layerType == "S2W":
            origin, neighbor = w, s
        elif self.layerType == "S2S":
            assert torch.equal(w, s)
            origin, neighbor = w, s
        else:
            origin, neighbor = None, None

        if self.layerType == "S2S":
            x = self.layer(g, neighbor)
            h = F.elu(x)
        else:
            x = self.layer(g, origin, neighbor)
            h = F.elu(x)
        h = h + origin
        h = self.ffn(h.unsqueeze(0)).squeeze(0)
        return h

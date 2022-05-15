''#!/usr/bin/python
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

import torch
import torch.nn as nn

from module.GATLayer import SGATLayer

######################################### StackLayer #########################################

class MultiHeadSGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, attn_drop_out, merge='cat'):
        super(MultiHeadSGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(
                SGATLayer(in_dim, out_dim))  # [n_nodes, hidden_size]
        self.merge = merge
        self.dropout = nn.Dropout(attn_drop_out)

    def forward(self, g, o, h):
        head_outs = [attn_head(g, self.dropout(h)) for attn_head in self.heads]  # n_head * [n_nodes, hidden_size]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs, dim=1)  # [n_nodes, hidden_size * n_head]
        else:
            # merge using average
            return torch.mean(torch.stack(head_outs))

class MultiHeadLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, attn_drop_out, feat_embed_size, layer, merge='cat'):
        super(MultiHeadLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(layer(in_dim, out_dim, feat_embed_size))  # [n_nodes, hidden_size]
        self.merge = merge
        self.dropout = nn.Dropout(attn_drop_out)

    def forward(self, g, o, h):
        head_outs = [attn_head(g, self.dropout(o), self.dropout(h)) for attn_head in self.heads]  # n_head * [n_nodes, hidden_size]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            result = torch.cat(head_outs, dim=1)  # [n_nodes, hidden_size * n_head]
        else:
            # merge using average
            result = torch.mean(torch.stack(head_outs))
        return result

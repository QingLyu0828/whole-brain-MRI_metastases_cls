import copy
import math

import torch
import torch.nn as nn

from torch.nn import Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from .feature_extraction import FeatureExtraction
from .feature_combination import DecoderBranch, SegmentationHead

class Attention(nn.Module):
    def __init__(self, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = 12
        self.attention_head_size = int(768 / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(768, self.all_head_size)
        self.key = Linear(768, self.all_head_size)
        self.value = Linear(768, self.all_head_size)

        self.out = Linear(768, 768)

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self):
        super(Mlp, self).__init__()
        self.fc1 = Linear(768, 768*2)
        # self.fc1 = Linear(768, 3072)
        self.fc2 = Linear(768*2, 768)
        self.act_fn = torch.nn.functional.gelu
        self.dropout = Dropout(0.1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        img_size = _pair(img_size)

        n_patches = int((img_size[0] / 32) * (img_size[1] / 32))  

        self.hybrid_model = FeatureExtraction()
        in_channels = self.hybrid_model.width * 32
        
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=768,
                                       kernel_size=1,
                                       stride=1)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, 768))

        self.dropout = Dropout(0.1)
        
        # self.conv1 = nn.Conv2d(4, 3, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        # print("input embed size:",x.size())

        # x = self.conv1(x)
        
        x, features = self.hybrid_model(x)

        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        # print("patch_embed size:",x.size())
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)
        # print("embed output size:",x.size())

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features


class Block(nn.Module):
    def __init__(self, vis):
        super(Block, self).__init__()
        self.hidden_size = 768
        self.attention_norm = LayerNorm(768, eps=1e-6)
        self.ffn_norm = LayerNorm(768, eps=1e-6)
        self.ffn = Mlp()
        self.attn = Attention(vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights


class Encoder(nn.Module):
    def __init__(self, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(768, eps=1e-6)
        num_layers = 12
        for _ in range(num_layers):
            layer = Block(vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            # print("encoder block size:",hidden_states.size())

            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(img_size=img_size)
        self.encoder = Encoder(vis)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        # print("trans output size:",encoded.size())

        return encoded, attn_weights, features


class VisionTransformer(nn.Module):
    def __init__(self, img_size=512, num_classes=2, vis=False):
        super(VisionTransformer, self).__init__()
        self.transformer = Transformer(img_size, vis) 
        self.decoder = DecoderBranch()
        self.segmentation_head = SegmentationHead(
            in_channels=32,
            out_channels=num_classes,
            kernel_size=3,
        )

    def forward(self, x):
        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        x = self.decoder(x, features)
        # print("decoder output size:",x.size())
        logits = self.segmentation_head(x)
        # print("segmentation_head size:",logits.size())        
        return logits



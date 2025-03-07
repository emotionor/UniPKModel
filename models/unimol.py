from __future__ import absolute_import, division, print_function

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from unicore.data import Dictionary
from unicore.models import BaseUnicoreModel
from unicore.modules import LayerNorm, init_bert_params
from unicore.utils import get_activation_fn

from utils import (DICT_PATH, PRE_TRAIN_WEIGHT_PATH, logger, pad_1d_tokens,
                     pad_2d, pad_coords)
from .nnmodelzoo import ClassificationHead, GaussianLayer, NonLinearHead
from .transformers import TransformerEncoderWithPair

BACKBONE = {
    'transformer': TransformerEncoderWithPair,
}

class UniMolModel(BaseUnicoreModel):
    def __init__(self, output_dim=2, pretrain='mol_pre_no_h_220816.pt', return_rep=False):
        super().__init__()
        self.args = base_architecture()
        self.output_dim = output_dim
        self.return_rep = return_rep
        self.pretrain_path = os.path.join(PRE_TRAIN_WEIGHT_PATH, pretrain)
        self.dictionary =  Dictionary.load(DICT_PATH)
        self.mask_idx = self.dictionary.add_symbol("[MASK]", is_special=True)
        self.padding_idx = self.dictionary.pad()
        self.embed_tokens = nn.Embedding(
            len(self.dictionary), self.args.encoder_embed_dim, self.padding_idx
        )
        self.encoder = BACKBONE[self.args.backbone](
            encoder_layers=self.args.encoder_layers,
            embed_dim=self.args.encoder_embed_dim,
            ffn_embed_dim=self.args.encoder_ffn_embed_dim,
            attention_heads=self.args.encoder_attention_heads,
            emb_dropout=self.args.emb_dropout,
            dropout=self.args.dropout,
            attention_dropout=self.args.attention_dropout,
            activation_dropout=self.args.activation_dropout,
            max_seq_len=self.args.max_seq_len,
            activation_fn=self.args.activation_fn,
            no_final_head_layer_norm=self.args.delta_pair_repr_norm_loss < 0,
        )
        K = 128
        n_edge_type = len(self.dictionary) * len(self.dictionary)
        self.gbf_proj = NonLinearHead(
            K, self.args.encoder_attention_heads, self.args.activation_fn
        )
        self.gbf = GaussianLayer(K, n_edge_type)
        self.classification_head = ClassificationHead(
            input_dim=self.args.encoder_embed_dim,
            inner_dim=self.args.encoder_embed_dim,
            num_classes=self.output_dim,
            activation_fn=self.args.pooler_activation_fn,
            pooler_dropout=self.args.pooler_dropout,
        )
        self.apply(init_bert_params)
        self.load_pretrained_weights(path=self.pretrain_path)

    def load_pretrained_weights(self, path):
        if path is not None:
            logger.debug("Loading pretrained weights from {}".format(path))
            state_dict = torch.load(path, map_location=lambda storage, loc: storage)
            if path == self.pretrain_path:
                self.load_state_dict(state_dict['model'], strict=False)
            else:
                self.load_state_dict(state_dict['model_state_dict'], strict=False)

    @classmethod
    def build_model(cls, args):
        """Build a new model instance."""
        return cls(args)

    def forward(
        self,
        src_tokens,
        src_distance,
        src_coord,
        src_edge_type,
        **kwargs
    ):
        padding_mask = src_tokens.eq(self.padding_idx)
        if not padding_mask.any():
            padding_mask = None
        x = self.embed_tokens(src_tokens)

        def get_dist_features(dist, et):
            n_node = dist.size(-1)
            gbf_feature = self.gbf(dist, et)
            gbf_result = self.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias

        graph_attn_bias = get_dist_features(src_distance, src_edge_type)
        (
            encoder_rep,
            _,
            _,
            _,
            _,
        ) = self.encoder(x, padding_mask=padding_mask, attn_mask=graph_attn_bias)
        if self.return_rep:
            return encoder_rep[:, 0, :]
        self.encoder_rep = encoder_rep
        logits = self.classification_head(encoder_rep)
        return logits

    def batch_collate_fn(self, samples):
        batch = {}
        for k in samples[0][0].keys():
            if k == 'src_coord':
                v = pad_coords([torch.tensor(s[0][k]).float() for s in samples], pad_idx=0.0)
            elif k == 'src_edge_type':
                v = pad_2d([torch.tensor(s[0][k]).long() for s in samples], pad_idx=self.padding_idx)
            elif k == 'src_distance':
                v = pad_2d([torch.tensor(s[0][k]).float() for s in samples], pad_idx=0.0)
            elif k == 'src_tokens':
                v = pad_1d_tokens([torch.tensor(s[0][k]).long() for s in samples], pad_idx=self.padding_idx)
            batch[k] = v
        label = torch.tensor([s[1] for s in samples]).float()
        return batch, label

def base_architecture():
    args = argparse.ArgumentParser()
    args.encoder_layers = getattr(args, "encoder_layers", 15)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 64)
    args.dropout = getattr(args, "dropout", 0.1)
    args.emb_dropout = getattr(args, "emb_dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.2)
    args.max_seq_len = getattr(args, "max_seq_len", 512)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.post_ln = getattr(args, "post_ln", False)
    args.backbone = getattr(args, "backbone", "transformer")
    args.delta_pair_repr_norm_loss = getattr(args, "delta_pair_repr_norm_loss", -1.0)
    return args
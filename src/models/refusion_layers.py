import re, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
logger = logging.getLogger(__name__)

from typing import List, Optional

COMMON_LAYERS_PATTERN = ["layers", "h", "block", "blocks"]

class ReFusionLayer(nn.Module):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        retriever, 
        fusion_strategy, 
        retrieve_texts,
        layer_name=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.retriever = retriever
        self.fusion_strategy = fusion_strategy
        self.retrieve_texts = retrieve_texts
        self.layer_name = layer_name
        

        # Initialize parameters to mask some retrievals
        self.mask = torch.Tensor(self.retriever.topk, 1)
        nn.init.xavier_uniform_(self.mask)
        self.mask = nn.Parameter(self.mask, requires_grad=True)

        # Initialize parameters for the ordered mask
        self.beta = torch.Tensor(self.retriever.topk, self.retriever.retrieval_dim)
        nn.init.xavier_uniform_(self.beta)
        self.beta = nn.Parameter(self.beta, requires_grad=True)
        self.lb = -5.0
        self.ub = 5.0
        self.tau = 5.0

    def update_layer(self, retriever, fusion_strategy, retrieve_texts, layer_name=None):
        self.retriever = retriever
        self.fusion_strategy = fusion_strategy
        self.retrieve_texts = retrieve_texts
        self.layer_name = layer_name
        self.to(self.weight.device)
    
    def _get_ordered_mask(self):
        clamped_beta = F.sigmoid(torch.clamp(self.beta, self.lb, self.ub))
        qz = torch.cumprod(clamped_beta, dim=0) * (1 - clamped_beta)
        sample = F.gumbel_softmax(qz, tau=self.tau, hard=False)
        ordered_mask = torch.flip(sample.cumsum(dim=0), dims=[0])
        return ordered_mask

    def _get_neighbors(self, x, pos=0):
        if self.retrieve_texts:
            neighbors = self.retriever.fetch_from_cache()
        else:
            queries = x[:, pos, :].squeeze(1).detach().cpu().numpy()
            neighbors = self.retriever.search(queries)
        if isinstance(neighbors, np.ndarray):
            neighbors = torch.tensor(neighbors).to(x.device)
        if neighbors.shape[1] == self.retriever.topk * 2:
            neighbors = (neighbors[:, :self.retriever.topk, :] + neighbors[:, self.retriever.topk:, :]) / 2
        return neighbors

    def _add_cls(self, x, y, cls_pos=0, mask=None):
        neighbors = self._get_neighbors(x, cls_pos)
        neighbors = mask * neighbors if mask != None else neighbors
        neighbors = torch.mean(neighbors, dim=1, keepdim=False)
        neighbors = neighbors.unsqueeze(1).repeat(1, y.shape[1], 1)
        neighbors[:, 0:cls_pos, :] = 0
        neighbors[:, cls_pos+1:, :] = 0
        result = y + neighbors
        return result

    def _mask_add_cls(self, x, y, cls_pos=0):
        mask = F.softmax(self.mask, dim=0)
        return self._add_cls(x, y, cls_pos=cls_pos, mask=mask)

    def _ordered_mask_add_cls(self, x, y, cls_pos=0):
        mask = self._get_ordered_mask()
        return self._add_cls(x, y, cls_pos=cls_pos, mask=mask)

    def _residual_add_cls(self, x, y, cls_pos=0, mask=None):
        neighbors = self._get_neighbors(x, cls_pos)
        neighbors = mask * neighbors if mask != None else neighbors
        neighbors = self.pert_output(neighbors)
        neighbors = torch.mean(neighbors, dim=1, keepdim=False)
        neighbors = neighbors.unsqueeze(1).repeat(1, y.shape[1], 1)
        neighbors[:, 0:cls_pos, :] = 0
        neighbors[:, cls_pos+1:, :] = 0
        result = y + neighbors
        return result

    def _mask_residual_add_cls(self, x, y, cls_pos=0):
        mask = F.softmax(self.mask, dim=0)
        return self._residual_add_cls(x, y, cls_pos, mask)

    def _ordered_mask_residual_add_cls(self, x, y, cls_pos=0):
        mask = self._get_ordered_mask()
        return self._residual_add_cls(x, y, cls_pos=cls_pos, mask=mask)

    def fuse_retrieval(self, x, y, cls_pos=0):
        if self.fusion_strategy == 'add_cls':
            result = self._add_cls(x, y, cls_pos=cls_pos)
        elif self.fusion_strategy == 'mask_add_cls':
            result = self._mask_add_cls(x, y, cls_pos=cls_pos)
        elif self.fusion_strategy == 'ordered_mask_add_cls':
            result = self._ordered_mask_add_cls(x, y, cls_pos=cls_pos)
        elif self.fusion_strategy == 'residual_add_cls':
            result = self._residual_add_cls(x, y, cls_pos=cls_pos)
        elif self.fusion_strategy == 'mask_residual_add_cls':
            result = self._mask_residual_add_cls(x, y, cls_pos=cls_pos)
        elif self.fusion_strategy == 'ordered_mask_residual_add_cls':
            result = self._ordered_mask_residual_add_cls(x, y, cls_pos=cls_pos)
        else:
            raise NotImplemented
        return result


class RetrievalLinear(ReFusionLayer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight,
        bias,
        retriever, 
        fusion_strategy, 
        retrieve_texts,
        layer_name=None,
    ):
        super().__init__(
            in_features=in_features, 
            out_features=out_features, 
            retriever=retriever,
            fusion_strategy=fusion_strategy,
            retrieve_texts=retrieve_texts,
            layer_name=layer_name,
        )
        self.in_features = in_features
        self.out_features = out_features
        self.weight = weight
        self.bias = bias

    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype
        if '_x' in self.fusion_strategy:
            y = self.fuse_retrieval(x, x)
            result = F.linear(y, self.weight, bias=self.bias)
        else:
            y = F.linear(x, self.weight, bias=self.bias)
            result = self.fuse_retrieval(x, y)
        result = result.to(previous_dtype)
        return result


class ReFusionLinear(ReFusionLayer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight,
        bias,
        retriever, 
        fusion_strategy, 
        retrieve_texts,
        candidates: Optional[List[str]] = None,
        layer_name=None,
    ):
        super().__init__(
            in_features=in_features, 
            out_features=out_features, 
            retriever=retriever,
            fusion_strategy=fusion_strategy,
            retrieve_texts=retrieve_texts,
            layer_name=layer_name,
        )

        self.candidates = candidates if candidates != None else ['identity']
        self.linears = nn.ModuleList([nn.Linear(in_features, out_features) for i in range(len(self.candidates))])
        for linear in self.linears:
            linear.weight = weight
            linear.bias = bias

        self.bilevel_weights = torch.Tensor(len(self.candidates), 1)
        nn.init.xavier_uniform_(self.bilevel_weights)
        self.bilevel_weights = nn.Parameter(self.bilevel_weights)
        self.bilevel_weights.requires_grad = False

    def __del__(self):
        pass
        # norm_arch_para = F.softmax(self.bilevel_weights, dim=0)
        # module_index = torch.argmax(norm_arch_para, dim=0)
        # module_name = self.candidates[module_index]
        # logger.info(f"Layer {self.layer_name}")
        # logger.info(f"Arch. Para. {norm_arch_para.detach().cpu().numpy().squeeze()}")
        # logger.info(f"Choose {module_name}")

    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype

        def module_forward(x, module_index, module_name, arch_weight):
            y = F.linear(x, self.linears[module_index].weight, bias=self.linears[module_index].bias)
            if module_name == 'identity':
                result = y * arch_weight
            elif module_name == 'add_cls':
                z = self._add_cls(x, y)
                result = z * arch_weight
            elif module_name == 'mask_add_cls':
                z = self._mask_add_cls(x, y)
                result = z * arch_weight
            elif module_name == 'ordered_mask_add_cls':
                z = self._ordered_mask_add_cls(x, y)
                result = z * arch_weight
            elif module_name == 'residual_add_cls':
                z = self._residual_add_cls(x, y)
                result = z * arch_weight
            elif module_name == 'mask_residual_add_cls':
                z = self._mask_residual_add_cls(x, y)
                result = z * arch_weight
            elif module_name == 'ordered_mask_residual_add_cls':
                z = self._ordered_mask_residual_add_cls(x, y)
                result = z * arch_weight
            else:
                result = torch.zeros_like(y).to(x.device)
            return result

        norm_arch_para = F.softmax(self.bilevel_weights, dim=0)
        if self.training:
            result = torch.zeros(x.shape[0], x.shape[1], self.out_features).to(x.device)
            for module_index, module_name in enumerate(self.candidates):
                result += module_forward(x, module_index, module_name, norm_arch_para[module_index])
        else:
            module_index = torch.argmax(norm_arch_para, dim=0)
            module_name = self.candidates[module_index]
            result = module_forward(x, module_index, module_name, 1)
        result = result.to(previous_dtype)
        return result

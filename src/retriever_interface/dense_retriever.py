import re
import sys
import torch
import numpy as np
from glob import glob
from sentence_transformers import SentenceTransformer
from retriever_interface.base_retriever import BaseRetriever
sys.path.append('/root/autodl-tmp/wsy/benchmark-ralm/lib/retriever-lib/src')
from faisslib.retriever import FaissRetriever
from models.refusion_layers import ReFusionLayer, ReFusionLinear, RetrievalLinear

COMMON_LAYERS_PATTERN = ["layers", "h", "block", "blocks"]

class DenseRetriever(BaseRetriever):
    def __init__(self, args):
        super(DenseRetriever, self).__init__(args)
        self.retriever = FaissRetriever(
            args.retriever_path, 
            args.nprobe, 
            args.topk, 
            args.retriever_device, 
            args.index_path
        )

    def save_in_cache(self, neighbors):
        self.retriever.save_in_cache(neighbors)
    
    def retrieve(self, query_texts):
        query_embs = self.encoder.encode(query_texts, show_progress_bar=False)
        results = self.retriever.search(query_embs)
        return results

    def load_encoder(self, encoder_path, device):
        self.encoder = SentenceTransformer(encoder_path).to(device)

    def replace_modules(self, model, config, ):
        self.enable_search = config.fusion_strategy == "search"
        self.search_candidates = config.search_candidates if self.enable_search else []
        self.fusion_strategy = config.fusion_strategy
        self.retrieve_texts = config.retrieve_texts

        is_target_modules_in_base_model = False
        key_list = [key for key, _ in model.named_modules()]
        for key in key_list:
            if not self._check_target_module_exists(config, key):
                continue

            is_target_modules_in_base_model = True
            parent, target, target_name = self._get_submodules(model, key)

            if isinstance(target, ReFusionLayer):
                target.update_layer(
                    self.retriever,
                    self.fusion_strategy,
                    self.retrieve_texts,
                    layer_name=key
                )
            else:
                new_module = self._create_new_module(target, key)
                self._replace_module(parent, target_name, new_module, target)

        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )
        # if not config.full_training:
        #     self.mark_only_pert_as_trainable()

    def _check_target_module_exists(self, config, key):
        if isinstance(config.target_modules, str):
            target_module_found = re.fullmatch(config.target_modules, key)
        else:
            target_module_found = any(key.split('.')[-1] == target_key for target_key in config.target_modules)
            is_using_layer_indexes = getattr(config, "layers_to_transform", None) is not None
            layer_indexing_pattern = getattr(config, "layers_pattern", None)

            if target_module_found and layer_indexing_pattern != None:
                layers_pattern = COMMON_LAYERS_PATTERN if layer_indexing_pattern is None else layer_indexing_pattern
                layers_pattern = [layers_pattern] if isinstance(layers_pattern, str) else layers_pattern

                for pattern in layers_pattern:
                    layer_index = re.match(pattern, key)
                    if layer_index is not None:
                        layer_index = int(layer_index.group(1))
                        if is_using_layer_indexes:
                            if isinstance(config.layers_to_transform, int):
                                target_module_found = layer_index == config.layers_to_transform
                            else:
                                target_module_found = layer_index in config.layers_to_transform
                        break
                    else:
                        target_module_found = False
        return target_module_found

    def _get_submodules(self, model, key):
        parent = model.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[-1]
        target = model.get_submodule(key)
        return parent, target, target_name

    def _create_new_module(self, target, layer_name=None):
        if isinstance(target, torch.nn.Linear):
            if self.enable_search:
                new_module = ReFusionLinear(
                    in_features=target.in_features, 
                    out_features=target.out_features, 
                    weight=target.weight,
                    bias=target.bias, 
                    retriever=self.retriever,
                    fusion_strategy=self.fusion_strategy,
                    retrieve_texts=self.retrieve_texts,
                    candidates=self.search_candidates,
                    layer_name=layer_name,
                )
            else:
                new_module = RetrievalLinear(
                    in_features=target.in_features, 
                    out_features=target.out_features, 
                    weight=target.weight,
                    bias=target.bias, 
                    retriever=self.retriever,
                    fusion_strategy=self.fusion_strategy,
                    retrieve_texts=self.retrieve_texts,
                    layer_name=layer_name,
                )
        else:
            raise ValueError(
                f"Target module {target} is not supported. "
                f"Currently, only `torch.nn.Linear` are supported."
            )

        return new_module

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        old_device = None
        for n, p in old_module.named_parameters():
            old_device = p.device
        if old_device != None:
            for name, module in new_module.named_modules():
                module.to(old_device)


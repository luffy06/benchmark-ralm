import random
import evaluate
import logging
import dataclasses
import torch
import transformers
import os
import math
import numpy as np

from tqdm import tqdm
from util.args import (
    ModelArguments, 
    DynamicDataTrainingArguments, 
    DynamicTrainingArguments, 
    RetrieverArguments
)
from transformers import HfArgumentParser

import sys
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoConfig, AutoTokenizer, EvalPrediction
from typing import Callable, Dict, Optional, List, Union, Any, Mapping, Tuple

from data.dataset import GLUEDataset
from data.processors import (
    num_labels_mapping, 
    output_modes_mapping, 
    compute_metrics_mapping
)
from util.args import models_mapping
from util.trainer import Trainer
from retriever_interface import AutoRetriever
from models.roberta_prompt import RobertaForPromptFinetuning

logger = logging.getLogger(__name__)

transformer_type_mapping = {
    'roberta': 'encoder-only',
    't5': 'encoder-decoder',
    'llama': 'decoder-only',
    'gemma': 'decoder-only'
}

models_mapping = {
    "roberta": RobertaForPromptFinetuning,
}

def data_collator_glue(features: List[Any]) -> Dict[str, Any]:
    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]
    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)
    if "input_texts" in first and first["input_texts"] is not None:
        batch["input_texts"] = [f["input_texts"] for f in features]
    if "neighbors" in first and first["neighbors"] is not None:
        batch["neighbors"] = torch.tensor(np.stack([f["neighbors"] for f in features]))
    if "neighbor_texts" in first and first["neighbor_texts"] is not None:
        batch["neighbor_texts"] = []
        for f in features:
            for text in f["neighbor_texts"]:
                batch['neighbor_texts'].append(text)
    
    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ("label", "label_ids", "input_texts", "neighbors", "neighbor_texts") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.tensor(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.tensor([f[k] for f in features])
    return batch


def setup_parallel():
    rank = torch.distributed.get_rank()
    local_rank = os.environ['LOCAL_RANK']
    master_addr = os.environ['MASTER_ADDR']
    master_port = os.environ['MASTER_PORT']
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    logger.info(f"Rank = {rank} is initialized in {master_addr}:{master_port}; local_rank = {local_rank}")
    torch.cuda.set_device(rank)
    return local_rank


def set_seed(seed):
    transformers.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    # Process command line arguments
    parser = HfArgumentParser((ModelArguments, DynamicDataTrainingArguments, DynamicTrainingArguments, RetrieverArguments))

    if len(sys.argv) == 2 and sys.azrgv[1].endswith(".json"):
        model_args, data_args, training_args, retriever_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, retriever_args = parser.parse_args_into_dataclasses()

    if training_args.no_train:
        training_args.do_train = False
    if training_args.no_predict:
        training_args.do_predict = False

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    # Setup CUDA, GPU & distributed training
    setup_parallel()
    set_seed(training_args.seed)

    logger.warning(
        "Process rank: %s, world size: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.world_size,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)
    logger.info("Training/evaluation parameters %s", training_args)

    # Get the GLUE task information
    try:
        num_labels = num_labels_mapping[data_args.task_name]
        output_mode = output_modes_mapping[data_args.task_name]
        logger.info("GLUE Task name: {}, number of labels: {}, output mode: {}".format(data_args.task_name, num_labels, output_mode))
    except KeyError:
        raise ValueError("GLUE Task not found: %s" % (data_args.task_name))

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        use_fast=False,
        cache_dir=model_args.cache_dir,
    )

    model = models_mapping[config.model_type].from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # Load retriever
    if retriever_args.retriever_type is not None:
        logger.info(f"Enable RAG, retriever type {retriever_args.retriever_type}, fusion mode {retriever_args.fusion_mode}")
        logger.info("Retrieval parameters %s", retriever_args)
        retriever = AutoRetriever.from_pretrained(args=retriever_args)
    else:
        retriever = None

    model_args.transformer_type = transformer_type_mapping[config.model_type]

    # Load datasets
    train_dataset = GLUEDataset(
        data_args, 
        tokenizer=tokenizer, 
        mode="train", 
        transformer_type=model_args.transformer_type, 
        retriever=retriever, 
    )
    eval_dataset = (
        GLUEDataset(
            data_args, 
            tokenizer=tokenizer, 
            mode="dev", 
            transformer_type=model_args.transformer_type, 
            retriever=retriever, 
        )
        if training_args.do_eval
        else None
    )
    test_dataset = (
        GLUEDataset(
            data_args, 
            tokenizer=tokenizer, 
            mode="test", 
            transformer_type=model_args.transformer_type, 
            retriever=retriever, 
        )
        if training_args.do_predict
        else None
    )

    # Pass dataset and argument information to the model
    if data_args.prompt:
        model.label_word_list = torch.tensor(train_dataset.label_word_list).long().cuda()
    model.model_args = model_args
    model.data_args = data_args
    model.tokenizer = tokenizer

    # Augment the model with the retriever
    if retriever_args.retriever_type != None:
        logger.info(f"Fusion mode {retriever_args.fusion_mode}")
        if retriever_args.fusion_mode == "concat":
            pass
        elif retriever_args.fusion_mode == "refusion":
            model = retriever.replace_modules(model, retriever_args)
        elif retriever_args.fusion_mode == "cross-attn":
            pass

    # Place the model on the GPU for distributed training
    model.to(training_args.device)
    if training_args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Build metric
    def build_compute_metrics_fn(task_name: str, transformer_type: str) -> Callable[[EvalPrediction], Dict]:
        def compute_metrics(p: EvalPrediction):
            # Note: the eval dataloader is sequential, so the examples are in order.
            # We average the logits over each sample for using demonstrations.
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            preds = np.squeeze(preds) if output_modes_mapping[task_name] == "regression" else np.argmax(preds, axis=1)
            result = compute_metrics_mapping[task_name](task_name=task_name, preds=preds, labels=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        return compute_metrics

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=build_compute_metrics_fn(data_args.task_name, model_args.transformer_type),
        tokenizer=tokenizer,
        data_collator=data_collator_glue,
    )

    # Training
    if training_args.do_train:
        if retriever_args.retriever_type != None:
            train_result = trainer.bilevel_train()
        else:
            train_result = trainer.train()
        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_predict:
        logger.info("  " + "***** Testing *****")
        logger.info("  Batch size = %d", training_args.eval_batch_size)
        model = model.module if hasattr(model, 'module') else model
        results = trainer.evaluate(eval_dataset=test_dataset)
        logger.info(f"result {results}")

        if data_args.task_name == "mnli":
            mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
            test_dataset = GLUEDataset(
                mnli_mm_data_args, 
                tokenizer=tokenizer, 
                mode="test", 
                transformer_type=model_args.transformer_type, 
                retriever=retriever, 
            )
            results = trainer.evaluate(eval_dataset=test_dataset)
            logger.info(f"result {results}")


if __name__ == "__main__":
    main()

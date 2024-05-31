import sys
import random
import logging
import dataclasses
import torch
import transformers
import os
import numpy as np

from tqdm import tqdm
from typing import Callable, Dict, List, Any, Mapping
from transformers import AutoConfig, AutoTokenizer, EvalPrediction

from data.dataset import (
    GLUEDataset,
    data_collator_with_retrievals,
)
from data.processors import (
    num_labels_mapping, 
    output_modes_mapping, 
    compute_metrics_mapping
)
from models.rag_model import ReFusionModel
from models.roberta_prompt import RobertaForPromptFinetuning
from util.trainer import Trainer
from util.func import (
    parse_system_arguments,
    setup_parallel,
    load_model_and_retriever,
)
from transformers.trainer_utils import set_seed

logger = logging.getLogger(__name__)

transformer_type_mapping = {
    "roberta": "encoder-only",
    "t5": "encoder-decoder",
    "llama": "decoder-only",
    "gemma": "decoder-only",
    "mistral": "decoder-only",
}

models_mapping = {
    "roberta": RobertaForPromptFinetuning,
    "t5": transformers.T5ForConditionalGeneration,
    "llama": transformers.LlamaForCausalLM,
    "gemma": transformers.GemmaForCausalLM,
    "mistral": transformers.MistralForCausalLM,
}

def main():
    # Parse command line arguments
    model_args, data_args, training_args, retriever_args = parse_system_arguments()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    # Setup CUDA, GPU & distributed training
    setup_parallel()
    set_seed(training_args.seed)

    logger.info(
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

    # Load pretrained model and tokenizer
    config, tokenizer, model, retriever = load_model_and_retriever(model_args=model_args, data_args=data_args, retriever_args=retriever_args, num_labels=num_labels)
    model_args.transformer_type = transformer_type_mapping[config.model_type]

    # Load datasets
    train_dataset = PPLDataset(
        data_args, 
        tokenizer=tokenizer, 
        mode="train", 
        transformer_type=model_args.transformer_type, 
        retriever=retriever, 
    )
    eval_dataset = (
        PPLDataset(
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
        PPLDataset(
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

    # Augment the model with the retriever
    if retriever_args.retriever_type != None:
        logger.info(f"Fusion mode {retriever_args.fusion_mode}")
        if retriever_args.fusion_mode == "concat":
            pass
        elif retriever_args.fusion_mode == "refusion":
            model = ReFusionModel(model, retriever, retriever_args)
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
        data_collator=data_collator_with_retrievals,
    )

    # Training
    if training_args.do_train:
        if retriever_args.retriever_type != None:
            logger.info("  " + "***** Bilevel Training *****")
            train_result = trainer.bilevel_train()
        else:
            logger.info("  " + "***** Training *****")
            train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)

    # Evaluation
    if training_args.do_predict:
        logger.info("  " + "***** Testing *****")
        logger.info("  Batch size = %d", training_args.eval_batch_size)
        logger.info("  Num steps = %d", len(test_dataset))
        results = trainer.evaluate(eval_dataset=test_dataset)
        logger.info("result %.4f" % results['eval_acc'])

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
            logger.info("result %.4f" % results['eval_acc'])


if __name__ == "__main__":
    main()

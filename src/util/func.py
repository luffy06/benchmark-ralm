import os
import sys
import logging
import torch
import transformers

from transformers import (
    HfArgumentParser,
    AutoConfig,
    AutoTokenizer,
)
from util.args import ModelArguments, DynamicDataTrainingArguments, DynamicTrainingArguments, RetrieverArguments
from models.roberta_prompt import RobertaForPromptFinetuning
from retriever_interface.base_retriever import load_retriever

logger = logging.getLogger(__name__)

models_mapping = {
    "roberta": RobertaForPromptFinetuning,
    "t5": transformers.T5ForConditionalGeneration,
    "llama": transformers.LlamaForCausalLM,
    "gemma": transformers.GemmaForCausalLM,
    "mistral": transformers.MistralForCausalLM,
}

def parse_system_arguments():
    # Process command line arguments
    parser = HfArgumentParser((ModelArguments, DynamicDataTrainingArguments, DynamicTrainingArguments, RetrieverArguments))

    if len(sys.argv) == 2 and sys.azrgv[1].endswith(".json"):
        model_args, data_args, training_args, retriever_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, retriever_args = parser.parse_args_into_dataclasses()
    return model_args, data_args, training_args, retriever_args


def setup_parallel():
    rank = torch.distributed.get_rank()
    local_rank = os.environ['LOCAL_RANK']
    master_addr = os.environ['MASTER_ADDR']
    master_port = os.environ['MASTER_PORT']
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    logger.info(f"Rank = {rank} is initialized in {master_addr}:{master_port}; local_rank = {local_rank}")
    torch.cuda.set_device(rank)
    return local_rank

def load_model_and_retriever(model_args, data_args, retriever_args, num_labels=None):
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
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

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
        retriever = load_retriever(args=retriever_args)
    else:
        retriever = None

    model.model_args = model_args
    model.data_args = data_args
    model.tokenizer = tokenizer

    return config, tokenizer, model, retriever

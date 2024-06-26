import logging

from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import TrainingArguments
from transformers import SchedulerType
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, List, Union

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    transformer_type: str = field(
        default='encoder'
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    # Few-shot type
    #   - finetune: standard fine-tuning
    #   - prompt: prompt-based fine-tuning
    few_shot_type: str = field(
        default='prompt',
        metadata={"help": "Few-shot learning model type. Choice: finetune, prompt"}
    )

    # Only for BERT-type model
    random_segment: bool = field(
        default=False,
        metadata={"help": "Whether to reinitialize the token type embeddings (only for BERT)."}
    )


@dataclass
class DynamicDataTrainingArguments(DataTrainingArguments):
    """
    Arguments for dynamic training.
    """
    num_k: Optional[int] = field(
        default=16,
        metadata={"help": "Number of training instances per class"}
    )

    # For prompting
    template: str = field(
        default=None,
        metadata={"help": "Template"}
    )

    mapping: str = field(
        default=None,
        metadata={"help": "Label word mapping"}
    )

    template_path: str = field(
        default=None,
        metadata={"help": "Path to a txt file that stores all the templates, one per line. Do not set this when prompt_path is used"}
    )

    mapping_path: str = field(
        default=None,
        metadata={"help": "Path to a txt file that stores all the label word mappings, one per line. Do not set this when prompt_path is used"}
    )

    prompt_path: str = field(
        default=None,
        metadata={"help": "Path to a txt file that stores all the prompts (templates and mappings), one per line"}
    )
 
    template_id: int = field(
        default=None,
        metadata={"help": "Template id if using template_path"}
    )

    mapping_id: int = field(
        default=None,
        metadata={"help": "Mapping id if using template_path"}
    )

    prompt_id: int = field(
        default=None,
        metadata={"help": "Prompt id if using prompt_path"}
    )

    top_n_template: int = field(
        default=None,
        metadata={"help": "Use top-n template in the template path"}
    )

    debug_mode: bool = field(
        default=False,
        metadata={"help": "Debug mode"}
    )

    # For max length
    first_sent_limit: int = field(
        default=None,
        metadata={"help": "Limit the length of the first sentence (i.e., sent_0)"}
    )

    other_sent_limit: int = field(
        default=None,
        metadata={"help": "Limit the length of sentences other than the first sentence"}
    )

    use_full_length: bool = field(
        default=None,
        metadata={"help": "Use the full length (512)"}
    )

    truncate_head: bool = field(
        default=False,
        metadata={"help": "When exceeding the maximum length, truncate the head instead of the tail."}
    )

    # Do not set up the following fields. They are set up automatically.
    prompt: bool = field(
        default=True,
        metadata={"help": "Whether to use prompt-based fine-tuning"}
    )
    template_list: List[str] = field(
        default=None,
        metadata={"help": "(DO NOT List of templates (only initialized after the program starts."}
    )
    return_texts: bool = field(
        default=False,
    )


@dataclass
class DynamicTrainingArguments(TrainingArguments):
    # For ensemble
    array_id: int = field(
        default=-1,
        metadata={"help": "Array ID (contains seed and hyper-paramter search) to idenfity the model"}
    )

    model_id: int = field(
        default=-1,
        metadata={"help": "Model ID (contains template information) to identify the model"}
    )

    save_logit: bool = field(
        default=False,
        metadata={"help": "Save test file logit with name $TASK-$MODEL_ID-$ARRAY_ID.npy"}
    )

    save_logit_dir: str = field(
        default=None,
        metadata={"help": "Where to save the prediction result"}
    )

    # Regularization
    fix_layers: int = field(
        default=0,
        metadata={"help": "Fix bottom-n layers when optimizing"}
    )

    # Training
    save_at_last: bool = field(
        default=True,
        metadata={"help": "Instead of saving the best (dev performance) checkpoint, save the last checkpoint"}
    )

    # Turn off train/test
    no_train: bool = field(
        default=False,
        metadata={"help": "No training"}
    )
    no_predict: bool = field(
        default=False,
        metadata={"help": "No test"}
    )

    # Arguments for bilevel optimization
    learning_rate_bilevel: float = field(
        default=5e-5, 
        metadata={"help": "The initial learning rate for AdamW."}
    )
    weight_decay_bilevel: float = field(
        default=0.0, 
        metadata={"help": "Weight decay for AdamW if we apply some."}
    )
    adam_beta1_bilevel: float = field(
        default=0.9, 
        metadata={"help": "Beta1 for AdamW optimizer"}
    )
    adam_beta2_bilevel: float = field(
        default=0.999, 
        metadata={"help": "Beta2 for AdamW optimizer"}
    )
    adam_epsilon_bilevel: float = field(
        default=1e-8, 
        metadata={"help": "Epsilon for AdamW optimizer."}
    )
    max_grad_norm_bilevel: float = field(
        default=1.0, 
        metadata={"help": "Max gradient norm."}
    )
    lr_scheduler_type_bilevel: Union[SchedulerType, str] = field(
        default="linear",
        metadata={"help": "The scheduler type to use."},
    )
    lr_scheduler_kwargs_bilevel: Optional[Dict] = field(
        default_factory=dict,
        metadata={
            "help": (
                "Extra parameters for the lr_scheduler such as {'num_cycles': 1} for the cosine with hard restarts"
            )
        },
    )
    warmup_steps_bilevel: int = field(
        default=0, 
        metadata={"help": "Linear warmup over warmup_steps."}
    )
    warmup_ratio_bilevel: float = field(
        default=0.0, 
        metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."}
    )
    # For generation length
    max_target_length: int = field(
        default=None,
    )


@dataclass
class RetrieverArguments:
    retriever_type: str = field(
        default=None,
        metadata={"help": "The choices include dense, sparse"},
    )
    fusion_mode: str = field(
        default=None,
        metadata={"help": "The choices include concat, refusion, cross-attn"},
    )
    encoder_path: str = field(
        default=None, 
        metadata={"help": "The path of encoder"}
    )
    retriever_device: int = field(
        default=-1,
    )
    retriever_path: str = field(
        default=None, 
        metadata={"help": "The instance of retrievers in different granularities"}
    )
    topk: int = field(
        default=1, 
        metadata={"help": "The number of retrieved neighbors"}
    )

    # Arguments for faiss retriever
    nprobe: int = field(
        default=500, 
        metadata={"help": "The number of probing clusters in faiss"}
    )
    index_path: str = field(
        default=None, 
        metadata={"help": "The path of index"}
    )

    # Arguments for fusion techniques
    # Arguments for concat

    # Arguments for refusion
    target_modules: List[str] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with PERT."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    layers_to_transform: List[int] = field(
        default=None,
        metadata={
            "help": "The layer indexes to transform, is this argument is specified, PEFT will transform only the layers indexes that are specified inside this list. If a single integer is passed, PEFT will transform only the layer at this index."
        },
    )
    layers_pattern: str = field(
        default=None,
        metadata={
            "help": "The layer pattern name, used only if `layers_to_transform` is different to None and if the layer pattern is not in the common layers pattern."
        },
    )
    fusion_strategy: str = field(
        default=None, 
        metadata={"help": "The fusion strategy to fuse the retrievals. The choices include rerank, ordered-mask, search"},
    )
    search_candidates: List[str] = field(
        default=None,
    )
    retrieve_texts: bool = field(
        default=False,
    )

    # Arguments for cross-attn

    # query_dim: int = field(
    #     default=0,
    #     metadata={"help": "The dimension of feature for query"}
    # )
    # fuse_dim: int = field(
    #     default=0,
    #     metadata={"help": "The dimension of feature for fusion"}
    # )
    # full_training: bool = field(
    #     default=False
    # )
    # load_nas: str = field(
    #     default=None, 
    #     metadata={"help": "The path of nas weights"}
    # )

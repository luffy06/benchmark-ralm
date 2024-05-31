import os
import torch
import logging
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from huggingface_hub import login
from datasets import load_dataset
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)


def dump_args(args, output_dir=None, output_file=None):
    assert output_dir is None or output_file is None

    logger.info(" **************** CONFIGURATION **************** ")
    for key, val in sorted(vars(args).items()):
        keystr = "{}".format(key) + (" " * (30 - len(key)))
        logger.info("%s -->   %s", keystr, val)
    logger.info(" **************** CONFIGURATION **************** ")

    if output_dir is not None or output_file is not None:
        output_file = output_file or os.path.join(output_dir, "args.txt")
        with open(output_file, "w") as f:
            for key, val in sorted(vars(args).items()):
                keystr = "{}".format(key) + (" " * (30 - len(key)))
                f.write(f"{keystr}   {val}\n")

def load_tokenizer(model_name):
    if "llama" in model_name:
        return LlamaTokenizer.from_pretrained(model_name)
    return AutoTokenizer.from_pretrained(model_name)


def load_model_and_tokenizer(model_name, model_parallelism=False, cache_dir=None, auth_token=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_count = torch.cuda.device_count()
    logger.info(f"Device: {device}, Device count: {device_count}")

    config = AutoConfig.from_pretrained(model_name)
    model_args = {}
    if cache_dir is not None:
        model_args["cache_dir"] = cache_dir
    if model_parallelism:
        model_args["device_map"] = "auto"
        model_args["low_cpu_mem_usage"] = True
    if hasattr(config, "torch_dtype") and config.torch_dtype is not None:
        model_args["torch_dtype"] = config.torch_dtype
    if auth_token is not None:
        model_args["use_auth_token"] = auth_token

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_args).eval()
    if not model_parallelism:
        model = model.to(device)
    tokenizer = load_tokenizer(model_name)

    if device_count > 1 and not model_parallelism:
        model = torch.nn.DataParallel(model)

    return model, tokenizer, config, device

def load_data(load_from, dataset_path, dataset_name, dataset_split):
    if load_from == "hf":
        dataset = load_dataset(dataset_path, dataset_name, split=dataset_split)
        dataset = "".join([x["text"] if x["text"] else " \n" for x in dataset])
    else:
        with open(dataset_path, "r") as f:
            dataset = f.read()
    return dataset

import os
import json
import logging
import argparse

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from utils import dump_args, load_model_and_tokenizer, load_data

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)


def get_retrieved_doc_ids(retrieved_example, tokenizer):
    docs = [retrieval["text"] for retrieval in retrieved_example["retrievals"]]
    docs = ' '.join(docs)
    doc_ids = tokenizer(docs, add_special_tokens=False, return_tensors="pt").input_ids
    return doc_ids

def eval_dataset(
        model,
        tokenizer,
        dataset,
        device,
        max_length,
        model_max_length,
        output_dir=None,
        stride=4,
        mask_id=-100,
        normalization_level="word",
        retrieval_dataset=None,
):
    encodings = tokenizer(dataset, add_special_tokens=False, return_tensors="pt")
    
    logger.info(f"Max model input length: {model_max_length}")
    logger.info(f"Max context length: {max_length}")
    logger.info(f"Stride:  {stride}")
    # Number of tokens in dataset
    dataset_len = encodings.input_ids.shape[1]
    logger.info(f"Dataset length: {dataset_len}")
    logger.info(f"Number of queries: {dataset_len // stride}")
    if retrieval_dataset != None:
        logger.info(f"Number of retrievals: {len(retrieval_dataset)}")
    if normalization_level == "word":
        counter = dataset.count(" ")
    elif normalization_level == "token":
        counter = dataset_len
    else:
        raise ValueError(f"Unknown normalization_level: '{normalization_level}'")

    logger.info(f"Normalization factor (num tokens/words..): {counter}")

    nlls = []
    prev_end_loc = 0

    all_token_ppls = []
    all_tokens_to_predict = []
    idx = 0
    for begin_loc in tqdm(range(0, dataset_len, stride)):
        end_loc = min(begin_loc + max_length, dataset_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)

        if retrieval_dataset is not None:
            assert idx < len(retrieval_dataset), f"Illegal idx {idx} of {len(retrieval_dataset)}"
            if len(retrieval_dataset[idx]["retrievals"]) > 0:
                retrieved_example = retrieval_dataset[idx]
                retrieved_doc_ids = get_retrieved_doc_ids(retrieved_example, tokenizer).to(device)
                input_ids = torch.cat((retrieved_doc_ids, input_ids), 1).to(device)
                input_ids = input_ids[:, -model_max_length:]
            
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = mask_id

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # Calculate per-token loss
            if trg_len < max_length:
                neg_log_likelihood = outputs.loss * trg_len
                lm_logits = outputs.logits[..., -trg_len-1:-1, :]
                labels = target_ids[..., -trg_len:]
            else:
                neg_log_likelihood = outputs.loss * (max_length - 1)
                lm_logits = outputs.logits[..., :-1, :]
                labels = target_ids[..., 1:]
            neg_log_likelihood = neg_log_likelihood.to(torch.float32).squeeze().cpu()
            lm_logits = lm_logits.to(torch.float32)

            loss_fct = CrossEntropyLoss(reduction="none")
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1)).cpu()
            token_ppls = loss.tolist()
            tokens_to_predict = labels.view(-1).cpu().tolist()

        nlls.append(neg_log_likelihood)
        all_token_ppls.append(token_ppls)
        all_tokens_to_predict.append(tokens_to_predict)
        assert len(all_token_ppls) == len(all_tokens_to_predict)

        prev_end_loc = end_loc
        idx += 1
        if end_loc == dataset_len:
            break

    assert retrieval_dataset is None or len(retrieval_dataset) == idx

    ppl = torch.exp(torch.stack(nlls).sum() / counter).item()
    logger.info(f"Perplexity: {ppl}")
    ppl_to_assert = np.exp(sum([sum(x) for x in all_token_ppls]) / counter)
    assert np.abs(ppl - ppl_to_assert) < 1e-3, f"{ppl:.3f}, {ppl_to_assert:.3f}"

    if output_dir is not None:
        d = {"eval_perplexity": ppl}
        with open(os.path.join(output_dir, "eval.json"), "w") as f:
            f.write(json.dumps(d) + "\n")


def main(args):
    if args.output_dir is not None:
        os.makedirs(args.output_dir)
    dump_args(args, output_dir=args.output_dir)

    model, tokenizer, config, device = load_model_and_tokenizer(
        args.model_name, model_parallelism=args.model_parallelism, cache_dir=args.cache_dir, auth_token=args.auth_token
    )

    # Model context size (e.g., 1024 for GPT-2)
    max_length = args.max_length
    model_max_length = config.n_positions if hasattr(config, "n_positions") else config.max_position_embeddings
    if max_length is None or max_length > model_max_length:
        max_length = model_max_length

    dataset = load_data(args.load_from, args.dataset_path, args.dataset_name, args.dataset_split)

    retrieval_dataset = None
    if args.retrieved_file is not None:
        with open(args.retrieved_file, "r") as f:
            retrieval_dataset = json.load(f)

    eval_dataset(
        model,
        tokenizer,
        dataset,
        device,
        max_length=max_length,
        model_max_length=model_max_length,
        output_dir=args.output_dir,
        stride=args.stride,
        mask_id=args.mask_id,
        normalization_level=args.normalization_level,
        retrieval_dataset=retrieval_dataset,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", type=str)

    # Dataset params
    parser.add_argument("--load_from", type=str, choices=["hf", "file"], default="hf")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default="test")
    parser.add_argument("--normalization_level", choices=["word", "token"], default="word")

    # Model params
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--mask_id", type=int, default=-100)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--model_parallelism", action="store_true")
    parser.add_argument("--auth_token", type=str, default=None)

    # retrieval params
    parser.add_argument("--retrieved_file", type=str, default=None)

    args = parser.parse_args()

    main(args)

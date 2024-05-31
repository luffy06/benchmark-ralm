import os
import sys
import json
import logging
import argparse
from tqdm import tqdm
from utils import dump_args, load_model_and_tokenizer, load_data
from retriever_interface.retriever_factory import add_retriever_args, get_retriever

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

def main(args):
    dump_args(args, output_file=args.output_file.replace(".json", ".args.txt"))

    logger.info("Loading tokenizer...")
    model, tokenizer, config, device = load_model_and_tokenizer(
        args.tokenizer_name, model_parallelism=args.model_parallelism, cache_dir=args.cache_dir, auth_token=args.auth_token
    )

    logger.info("Loading dataset...")
    dataset = load_data(args.load_from, args.dataset_path, args.dataset_name, args.dataset_split)

    encodings = tokenizer(dataset, add_special_tokens=False, return_tensors="pt")
    dataset_len = encodings.input_ids.shape[1]
    logger.info(f"Dataset length: {dataset_len}")

    logger.info(f"Creating retriever of type {args.retrieval_type}...")
    retriever = get_retriever(args, tokenizer)

    # Ref: https://huggingface.co/docs/transformers/perplexity
    data = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, dataset_len, args.stride)):
        end_loc = min(begin_loc + args.max_length, dataset_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = args.mask_id

        d = {
            "begin_location": begin_loc,
            "end_location": end_loc,
            "target_length": trg_len,
        }
        data.append(d)

        prev_end_loc = end_loc
        if end_loc == dataset_len:
            break

    for i in range(0, len(data), args.batch_size):
        if i % 1000 == 0:
            logger.info(f"Finished processing {i}/{len(data)} strides")
        retriever.retrieve(encodings.input_ids, data[i : i + args.batch_size], k=args.topk)

    logger.info(f"Finished processing {len(data)}/{len(data)} strides")
    logger.info(f"Writing to {args.output_file}")
    with open(args.output_file, "w") as f:
        f.write(json.dumps(data, indent=4))
        f.write("\n")

    logger.info("Done!")


if __name__ == '__main__':
    assert sys.argv[1] == "--retrieval_type"
    retrieval_type = sys.argv[2]

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", required=True, type=str)

    # Dataset params
    parser.add_argument("--load_from", type=str, choices=["hf", "file"], default="hf")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default="test")

    # Model params
    parser.add_argument("--tokenizer_name", type=str, default="gpt2")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--mask_id", type=int, default=-100)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--model_parallelism", action="store_true")
    parser.add_argument("--auth_token", type=str, default=None)

    # Retrieval params
    parser.add_argument("--retrieval_type", required=True)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--topk", type=int, default=1)
    add_retriever_args(parser, retrieval_type)

    args = parser.parse_args()
    main(args)

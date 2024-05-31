import os
import argparse
from datasets import load_dataset

def download_dataset(dataset_name, output_dir, dataset_config=None):
    output_path = os.path.join(output_dir, dataset_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if dataset_name == "wmt16":
        dataset = load_dataset(dataset_name, dataset_config["language_pair"], token=dataset_config["token"])
        dataset_name = f"{dataset_name}_{dataset_config['language_pair']}"
    else:
        dataset = load_dataset(dataset_name, token=dataset_config["token"])

    if "train" in dataset:
        dataset["train"].to_json(os.path.join(output_path, f"{dataset_name}_train.json"))
    if "validation" in dataset:
        dataset["validation"].to_json(os.path.join(output_path, f"{dataset_name}_valid.json"))
    if "test" in dataset:
        dataset["test"].to_json(os.path.join(output_path, f"{dataset_name}_test.json"))
  

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to download")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the dataset")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face API token")
    parser.add_argument("--language_pair", type=str, default="tr-en", help="Language pair for the dataset")
    args = parser.parse_args()

    dataset_config = {
        "token": args.token,
        "language_pair": args.language_pair
    }

    download_dataset(args.dataset_name, args.output_dir, dataset_config)
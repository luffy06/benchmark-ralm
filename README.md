# benchmark-ralm

## Benchmark the In-Context Capability of Large Language Models

The workflow of this benchmarking include two steps,
1. Prepare the retrievals for the dataset.
2. Evaluate the in-context capability of large language models with retrievals on different tasks.

### Prepare Retrievals

We provide three types of retrievers, exact retriever, dense retriever, and sparse retriever. Specifically,
**Exact retriever.** This kind of retriever will retrieve the exact answer, and concat the query and answer with the prompt.
  
To prepare retrievals with exact retriever, you can run the following codes,

```bash
CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 \
  python src/in-context/prepare_retrieval_data.py \
    --retrieval_type exact \
    --output_file retrievals/llama-7b-exact.txt \
    --dataset_path wikitext \
    --dataset_name wikitext-103-v1 \
    --dataset_split test \
    --tokenizer_name llama-7b \
    --max_length 1024 \
    --stride 4 \
    --mask_id -100 \
    --batch_size 10000 \
    --topk 1
```

**Dense retriever.** This kind of retriever will retrieve the similar context over a dense embedding space.

To prepare retrievals with dense retriever, you can run the following codes,

```bash
CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 \
  python src/in-context/prepare_retrieval_data.py \
    --retrieval_type dense \
    --output_file retrievals/llama-7b-100K.txt \
    --dataset_path wikitext \
    --dataset_name wikitext-103-v1 \
    --dataset_split test \
    --tokenizer_name llama-7b \
    --max_length 1024 \
    --stride 4 \
    --mask_id -100 \
    --batch_size 1024 \
    --topk $TOPK \
    --encoder_name bert-base-uncased \
    --retriever_dir $RETRIEVER_DIR \
    --corpus_size 100K \
    --nprobe 512 \
    --device_id 0
```
, where `$RETRIEVER_DIR` is the directory path of the pre-built dense retriever.

### Evaluation

We provide various tasks to evaluate the in-context capability of large language models, such as language modeling, question-answer.

**Language modeling.** This evaluation computes model's perplexity on a given text.

To compute the perplexity, you can run the following codes,
```bash
CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 \
  python src/in-context/eval_ppl.py \
    --output_dir $OUTPUT_DIR \
    --dataset_path wikitext \
    --dataset_name wikitext-103-v1 \
    --dataset_split test \
    --model_name llama-7b \
    --max_length 1024 \
    --stride 4 \
    --mask_id -100
```

To compute the perplexity using retrieval augmentations, you can run the following codes,
```bash
CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 \
  python src/in-context/eval_ppl.py \
    --output_dir $OUTPUT_DIR \
    --dataset_path wikitext \
    --dataset_name wikitext-103-v1 \
    --dataset_split test \
    --model_name llama-7b \
    --max_length 1024 \
    --stride 4 \
    --mask_id -100 \
    --retrieved_file retrievals/llama-7b-exact.txt
```

### Pipeline Scripts

We provide scripts to pipeline the whole process.

To evaluate the in-context capability of llm on language modeling task, you can run the following script,
```bash
bash scripts/eval_ppc_ic.sh
```
You can modify the following hyper-parameters for different setting,
* `PROJECT_DIR`: the root path of this project.
* `DEVICE`: the device id to run the large language models.
* `RETRIEVER`: the retriever type,
  * `none`: no retriever;
  * `exact`: exact retriever;
  * `dense`, `dense-100k`, `dense-500k`, ...: dense retrievers (with different retrieval database size);
* `MASK_ID`: mask token id for masking labels;
* `MODEL`: large language model name;
* `MODEL_PATH`: the path of large language model;
* `OUTPUT_DIR`: the output directory path;
* `RETRIEVAL_DIR`: the path to storing retrievals;
* `BATCH_SIZE`: the batch size during preparing retrievals;
* `TOPK`: the number of retrievals per query;
* `ENCODER`: the encoder model name (only for dense retriever);
* `RETRIEVER_DIR`: the path of pre-built dense retrievers (only for dense retriever);
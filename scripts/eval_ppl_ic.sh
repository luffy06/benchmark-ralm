PROJECT_DIR=/home/wsy/Project/benchmark-ralm
DEVICE=0
RETRIEVER=none # none, exact, dense-100k, dense-500k, dense-1m, dense-2m, dense-10m
MASK_ID=-100
MODEL=opt-1.3b
MODEL_PATH=/mnt/wsy/models/$MODEL
OUTPUT_DIR=$PROJECT_DIR/outputs/$MODEL-$RETRIEVER
RETRIEVAL_DIR=$PROJECT_DIR/retrievals
BATCH_SIZE=1024
TOPK=1
RETRIEVER_LIB=$PROJECT_DIR/src/retriever-lib

if [[ -d $OUTPUT_DIR ]]; then
  rm -rf $OUTPUT_DIR
fi

if [[ ! -d $RETRIEVAL_DIR ]]; then
  mkdir $RETRIEVAL_DIR
fi

if [[ -f $RETRIEVAL_DIR/$MODEL-$RETRIEVER.txt ]]; then
  rm $RETRIEVAL_DIR/$MODEL-$RETRIEVER.txt
fi

if [[ $RETRIEVER == "exact" ]]; then
  # Idea case of exact search
  CUDA_VISIBLE_DEVICES=$DEVICE CUDA_LAUNCH_BLOCKING=1 \
    python $PROJECT_DIR/src/in-context/prepare_retrieval_data.py \
      --retrieval_type exact \
      --output_file $RETRIEVAL_DIR/$MODEL-$RETRIEVER.txt \
      --dataset_path wikitext \
      --dataset_name wikitext-103-v1 \
      --dataset_split test \
      --tokenizer_name $MODEL_PATH \
      --max_length 1024 \
      --stride 4 \
      --mask_id $MASK_ID \
      --batch_size 10000 \
      --topk $TOPK
elif [[ $RETRIEVER == dense* ]]; then
  ENCODER=/disk3/xy/LM/bert-base-uncased
  RETRIEVER_DIR=$RETRIEVER_LIB/metadata/wikitext-103-all/
  # Use a dense retriever to search
  CUDA_VISIBLE_DEVICES=$DEVICE CUDA_LAUNCH_BLOCKING=1 \
    python $PROJECT_DIR/src/in-context/prepare_retrieval_data.py \
      --retrieval_type dense \
      --output_file $RETRIEVAL_DIR/$MODEL-$RETRIEVER.txt \
      --dataset_path wikitext \
      --dataset_name wikitext-103-v1 \
      --dataset_split test \
      --tokenizer_name $MODEL_PATH \
      --max_length 1024 \
      --stride 4 \
      --mask_id $MASK_ID \
      --batch_size $BATCH_SIZE \
      --topk $TOPK \
      --encoder_name $ENCODER \
      --retriever_dir $RETRIEVER_DIR \
      --corpus_size $RETRIEVER \
      --nprobe 512 \
      --device_id 0
elif [[ $RETRIEVER == "openai" ]]; then
  PROMPT="I will give you an incomplete sentence, please predict the next %d words. You only need to return the completed sentence."
  OPENAI_MODEL="gpt-4"
  
  CUDA_VISIBLE_DEVICES=$DEVICE CUDA_LAUNCH_BLOCKING=1 \
    python $PROJECT_DIR/src/in-context/prepare_retrieval_data.py \
      --retrieval_type exact \
      --output_file $RETRIEVAL_DIR/$MODEL-$RETRIEVER.txt \
      --dataset_path wikitext \
      --dataset_name wikitext-103-v1 \
      --dataset_split test \
      --tokenizer_name $MODEL_PATH \
      --max_length 1024 \
      --stride 4 \
      --mask_id $MASK_ID \
      --batch_size 10000 \
      --topk $TOPK \
      --system_prompt $PROMPT \
      --model_name $OPENAI_MODEL
fi

if [[ $RETRIEVER == "none" ]]; then
  CUDA_VISIBLE_DEVICES=$DEVICE CUDA_LAUNCH_BLOCKING=1 \
    python $PROJECT_DIR/src/in-context/eval_ppl.py \
      --output_dir $OUTPUT_DIR \
      --dataset_path wikitext \
      --dataset_name wikitext-103-v1 \
      --dataset_split test \
      --model_name $MODEL_PATH \
      --max_length 1024 \
      --stride 4 \
      --mask_id $MASK_ID
else
  CUDA_VISIBLE_DEVICES=$DEVICE CUDA_LAUNCH_BLOCKING=1 \
    python $PROJECT_DIR/src/in-context/eval_ppl.py \
      --output_dir $OUTPUT_DIR \
      --dataset_path wikitext \
      --dataset_name wikitext-103-v1 \
      --dataset_split test \
      --model_name $MODEL_PATH \
      --max_length 1024 \
      --stride 4 \
      --mask_id $MASK_ID \
      --retrieved_file $RETRIEVAL_DIR/$MODEL-$RETRIEVER.txt
fi
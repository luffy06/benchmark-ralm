# Required environment variables:
# DEVICE: the device id for running models
# PORT: the port
# TYPE: finetune / prompt
# TASK: SST-2 / sst-5 / mr / cr / mpqa / subj / trec / CoLA / MNLI / SNLI / QNLI / RTE / MRPC / QQP / STS-B
# BS: batch size (recommendation: 2 / 4 / 8)
# LR: learning rate (recommendation: 1e-5 / 2e-5 / 5e-5)
# SEED: random seed (13 / 21 / 42 / 87 / 100)
# MODEL: pre-trained model name (roberta-*, bert-*), see Transformers model list

PROJECT_DIR=$(dirname "$(dirname "$(realpath "$0")")")
DEVICE=0,2
PORT=7777
TYPE=prompt
TASK_LIST=('SST-2') # 'sst-5' 'mr' 'cr' 'mpqa' 'subj' 'trec' 'CoLA' 'MNLI' 'SNLI' 'QNLI' 'RTE' 'MRPC' 'QQP')
BS=32
LR=1e-5
ARCH_LR=5e-5
SEED_LIST=(13) # 21 42 87 100)
MODEL=/root/autodl-tmp/wsy/models/roberta-large
IFS='/' read -ra ADDR <<< "$MODEL"
MODEL_NAME=${ADDR[-1]}

# Number of training instances per label
K=16

# Training steps
MAX_STEP=20

# Validation steps
EVAL_STEP=20

# Gradient accumulation steps
# For medium-sized GPUs (e.g., 2080ti with 10GB memory), they can only take 
# a maximum batch size of 2 when using large-size models. So we use gradient
# accumulation steps to achieve the same effect of larger batch sizes.
REAL_BS=16
GS=$(expr $BS / $REAL_BS)

for SEED in ${SEED_LIST[*]}
do
    for TASK in ${TASK_LIST[*]}
    do
        # Task specific parameters
        # The default length is 128 and the default number of samples is 16.
        # For some tasks, we use longer length or double demo (when using demonstrations, double the maximum length).
        # For some tasks, we use smaller number of samples to save time (because of the large size of the test sets).
        # All those parameters are set arbitrarily by observing the data distributions.
        TASK_EXTRA=""
        case $TASK in
            CoLA)
                TEMPLATE=*cls**sent_0*_This_is*mask*.*sep+*
                MAPPING="{'0':'incorrect','1':'correct'}"
                ;;
            SST-2)
                TEMPLATE=*cls**sent_0*_It_was*mask*.*sep+*
                MAPPING="{'0':'terrible','1':'great'}"
                ;;
            MRPC)
                TEMPLATE=*cls**sent_0**mask*,*+sentl_1**sep+*
                MAPPING="{'0':'No','1':'Yes'}"
                ;;
            QQP)
                TEMPLATE=*cls**sent_0**mask*,*+sentl_1**sep+*
                MAPPING="{'0':'No','1':'Yes'}"
                ;;
            STS-B)
                TEMPLATE=*cls**sent_0**mask*,*+sentl_1**sep+*
                MAPPING="{'0':'No','1':'Yes'}"
                ;;
            MNLI)
                TEMPLATE=*cls**sent-_0*?*mask*,*+sentl_1**sep+*
                MAPPING="{'contradiction':'No','entailment':'Yes','neutral':'Maybe'}"
                TASK_EXTRA="--max_seq_len 256"
                ;;
            SNLI)
                TEMPLATE=*cls**sent-_0*?*mask*,*+sentl_1**sep+*
                MAPPING="{'contradiction':'No','entailment':'Yes','neutral':'Maybe'}"
                TASK_EXTRA="--max_seq_len 256"
                ;;
            QNLI)
                TEMPLATE=*cls**sent-_0*?*mask*,*+sentl_1**sep+*
                MAPPING="{'not_entailment':'No','entailment':'Yes'}"
                ;;
            RTE)
                TEMPLATE=*cls**sent-_0*?*mask*,*+sentl_1**sep+*
                MAPPING="{'not_entailment':'No','entailment':'Yes'}"
                TASK_EXTRA="--max_seq_len 256 --first_sent_limit 240"
                ;;
            mr)
                TEMPLATE=*cls**sent_0*_It_was*mask*.*sep+*
                MAPPING="{0:'terrible',1:'great'}"
                TASK_EXTRA="--first_sent_limit 110 --other_sent_limit 50"
                ;;
            sst-5)
                TEMPLATE=*cls**sent_0*_It_was*mask*.*sep+*
                MAPPING="{0:'terrible',1:'bad',2:'okay',3:'good',4:'great'}"
                TASK_EXTRA="--first_sent_limit 110 --other_sent_limit 20"
                ;;
            subj)
                TEMPLATE=*cls**sent_0*_This_is*mask*.*sep+*
                MAPPING="{0:'subjective',1:'objective'}"
                TASK_EXTRA="--first_sent_limit 110 --other_sent_limit 50"
                ;;
            trec)
                TEMPLATE="*cls**mask*:*+sent_0**sep+*"
                MAPPING="{0:'Description',1:'Entity',2:'Expression',3:'Human',4:'Location',5:'Number'}"
                TASK_EXTRA="--first_sent_limit 110"
                ;;
            cr)
                TEMPLATE=*cls**sent_0*_It_was*mask*.*sep+*
                MAPPING="{0:'terrible',1:'great'}"
                TASK_EXTRA="--first_sent_limit 110 --other_sent_limit 50"
                ;;
            mpqa)
                TEMPLATE=*cls**sent_0*_It_was*mask*.*sep+*
                MAPPING="{0:'terrible',1:'great'}"
                TASK_EXTRA="--first_sent_limit 110 "
                ;;

        esac

        # Use a random number to distinguish different trails (avoid accidental overwriting)
        TRIAL_IDTF=$RANDOM
        DATA_DIR=$PROJECT_DIR/dataset/k-shot/$TASK/$K-$SEED
        LOG_DIR=$PROJECT_DIR/results/glue/$MODEL_NAME/$TASK

        if [[ ! -d $LOG_DIR ]]; then
            mkdir -p $LOG_DIR
        fi

        LOCAL_RANK=0,1 CUDA_VISIBLE_DEVICES=$DEVICE \
            python -m torch.distributed.launch \
                --nproc_per_node=2 \
                --use-env \
                --master_port $PORT $PROJECT_DIR/src/run_glue.py \
                --task_name $TASK \
                --data_dir $DATA_DIR \
                --overwrite_output_dir \
                --do_train \
                --do_eval \
                --do_predict \
                --evaluation_strategy steps \
                --model_name_or_path $MODEL \
                --few_shot_type $TYPE \
                --num_k $K \
                --max_seq_length 128 \
                --per_device_train_batch_size $REAL_BS \
                --per_device_eval_batch_size 16 \
                --gradient_accumulation_steps $GS \
                --learning_rate $LR \
                --max_steps $MAX_STEP \
                --logging_steps $EVAL_STEP \
                --eval_steps $EVAL_STEP \
                --output_dir checkpoints/$TASK-$TYPE-$K-$SEED-$MODEL_NAME-$TRIAL_IDTF \
                --seed $SEED \
                --template $TEMPLATE \
                --mapping $MAPPING \
                $TASK_EXTRA \
                > $LOG_DIR/$TYPE-$K-$SEED-$MODEL_NAME.log 2>&1
        rm -rf $PROJECT_DIR/checkpoints/$TASK-$TYPE-$K-$SEED-$MODEL_NAME-$TRIAL_IDTF
    done
done
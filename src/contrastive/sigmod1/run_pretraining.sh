MODEL=$1
CHECKPOINT=$2
BATCH=$3
LR=$4
TEMP=$5
EPOCH=$6
GPU=$7
CUDA_VISIBLE_DEVICES=$GPU python run_pretraining.py \
    --do_train \
	--dataset_name=sigmod1 \
    --train_file ../../data/processed/blocking-sigmod-1/blocking-sigmod-1-train.pkl.gz \
	--tokenizer=$MODEL \
	--grad_checkpoint=$CHECKPOINT \
    --output_dir ../../reports/contrastive/blocking-sigmod-1-$BATCH-$LR-$TEMP-$EPOCH-${MODEL##*/}/ \
	--temperature=$TEMP \
	--per_device_train_batch_size=$BATCH \
	--learning_rate=$LR \
	--weight_decay=0.01 \
	--num_train_epochs=$EPOCH \
	--lr_scheduler_type="linear" \
	--warmup_ratio=0.05 \
	--max_grad_norm=1.0 \
	--fp16 \
	--disable_tqdm=True \
	--save_strategy="epoch" \
	--logging_strategy="epoch" \
	#--dataloader_num_workers=8 \
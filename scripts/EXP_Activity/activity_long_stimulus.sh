#  sh scripts/EXP_Activity/activity_long_stimulus.sh
# add --individual for DLinear-I
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/ActivityLongStimulus" ]; then
    mkdir ./logs/ActivityLongStimulus
fi
pred_len=8
seq_len=32
label_len=4
root_path=/cs/student/projects1/aibh/2024/gcosta/mpci_data/
data_path=session_5ea6bb9b-6163-4e8a-816b-efe7002666b0.h5

# # Run Naive baseline (true null model)
python -u run_stat.py \
  --is_training 1 \
  --model Naive \
  --data ActivityStimulus \
  --root_path $root_path \
  --data_path $data_path \
  --model_id Naive_$seq_len'_'$pred_len \
  --features M \
  --label_len $label_len \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --des 'Exp' \
  --batch_size 128 \
  --itr 1  | tee logs/ActivityLongStimulus/Naive'_'$seq_len'_'$pred_len.log

# Run DummyLinear baseline
python -u run_stimulus.py \
  --is_training 1 \
  --model DummyLinear \
  --data ActivityStimulus \
  --root_path $root_path \
  --data_path $data_path \
  --model_id DummyLinear_$seq_len'_'$pred_len \
  --features M \
  --train_epochs 10\
  --label_len $label_len \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 5011 \
  --c_out 5000 \
  --loss mae \
  --des 'Exp' \
  --batch_size 128 \
  --learning_rate 0.01 \
  --itr 1  | tee logs/ActivityLongStimulus/DummyLinear'_'$seq_len'_'$pred_len.log


# TSMixer
python -u run_stimulus.py \
  --is_training 1 \
  --model TSMixer \
  --data ActivityStimulus \
  --root_path $root_path \
  --data_path $data_path \
  --model_id TSMixer_$seq_len'_'$pred_len \
  --features M \
  --label_len $label_len \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 5011 \
  --c_out 5000 \
  --loss mae \
  --des 'Exp' \
  --num_workers 5 \
  --train_epochs 10 \
  --patience 3 \
  --batch_size 8 \
  --learning_rate 0.01 \
  --itr 1  | tee logs/ActivityLongStimulus/TSMixer'_'$seq_len'_'$pred_len.log

# Linear
python -u run_stimulus.py \
  --is_training 1 \
  --model Linear \
  --data ActivityStimulus \
  --root_path $root_path \
  --data_path $data_path \
  --model_id Linear_$seq_len'_'$pred_len \
  --features M \
  --label_len $label_len \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 5011 \
  --c_out 5000 \
  --loss mae \
  --des 'Exp' \
  --num_workers 5 \
  --train_epochs 10 \
  --patience 3 \
  --batch_size 8 \
  --learning_rate 0.01 \
  --itr 1  | tee logs/ActivityLongStimulus/Linear'_'$seq_len'_'$pred_len.log


# DLinear
python -u run_stimulus.py \
  --is_training 1 \
  --model DLinear \
  --data ActivityStimulus \
  --root_path $root_path \
  --data_path $data_path \
  --model_id DLinear_$seq_len'_'$pred_len \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 5011 \
  --c_out 5000 \
  --loss mae \
  --des 'Exp' \
  --num_workers 5 \
  --train_epochs 10 \
  --patience 3 \
  --batch_size 8 \
  --learning_rate 0.01 \
  --itr 1  | tee logs/ActivityLongStimulus/DLinear'_'$seq_len'_'$pred_len.log

# Informer
python -u run_stimulus.py \
  --is_training 1 \
  --model Informer \
  --data ActivityStimulus \
  --root_path $root_path \
  --data_path $data_path \
  --model_id Informer_$seq_len'_'$pred_len \
  --features M \
  --label_len $label_len \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 5011 \
  --dec_in 5011 \
  --c_out 5000 \
  --loss mae \
  --des 'Exp' \
  --train_epochs 10 \
  --batch_size 32 \
  --freq m \
  --learning_rate 0.01 \
  --itr 1  | tee logs/ActivityLongStimulus/Informer'_'$seq_len'_'$pred_len.log


# Transformer
python -u run_stimulus.py \
  --is_training 1 \
  --model Transformer \
  --data ActivityStimulus \
  --root_path $root_path \
  --data_path $data_path \
  --model_id Transformer_$seq_len'_'$pred_len \
  --features M \
  --label_len $label_len \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 5011 \
  --dec_in 5011 \
  --c_out 5000 \
  --freq m \
  --loss mae \
  --des 'Exp' \
  --batch_size 32 \
  --learning_rate 0.01 \
  --itr 1  | tee logs/ActivityLongStimulus/Transformer'_'$seq_len'_'$pred_len.log

# Autoformer
python -u run_stimulus.py \
  --is_training 1 \
  --model Autoformer \
  --data ActivityStimulus \
  --root_path $root_path \
  --data_path $data_path \
  --model_id Autoformer_$seq_len'_'$pred_len \
  --features M \
  --train_epochs 50 \
  --label_len $label_len \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 5011 \
  --dec_in 5011 \
  --c_out 5000 \
  --freq m \
  --loss mae \
  --des 'Exp' \
  --batch_size 32 \
  --learning_rate 0.01 \
  --itr 1  | tee logs/ActivityLongStimulus/Autoformer'_'$seq_len'_'$pred_len.log

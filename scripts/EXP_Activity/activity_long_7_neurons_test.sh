#  sh scripts/EXP_Activity/activity_long_7_neurons_test.sh
# add --individual for DLinear-I
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/ActivityLong7NeuronsTest" ]; then
    mkdir ./logs/ActivityLong7NeuronsTest
fi
batch_size=8
pred_len=8
seq_len=16
label_len=8
experiment_name=ActivityLong7NeuronsTest
root_path=/cs/student/projects1/aibh/2024/gcosta/mpci_data/
data_path=session_5ea6bb9b-6163-4e8a-816b-efe7002666b0_7.h5

# # Run Naive baseline (true null model)
python -u run_stat.py \
  --is_training 1 \
  --model Naive \
  --data Activity \
  --root_path $root_path \
  --data_path $data_path \
  --model_id Naive_$seq_len'_'$pred_len \
  --features M \
  --label_len $label_len \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --des 'Exp' \
  --batch_size $batch_size \
  --itr 1  | tee logs/ActivityLong7NeuronsTest/Naive'_'$seq_len'_'$pred_len.log

# TSMixer
python -u run_longExp.py \
  --is_training 1 \
  --model TSMixer \
  --data Activity \
  --root_path $root_path \
  --data_path $data_path \
  --model_id TSMixer_$seq_len'_'$pred_len \
  --features M \
  --label_len $label_len \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 7 \
  --c_out 7 \
  --loss mae \
  --des 'Exp' \
  --experiment_name $experiment_name \
  --num_workers 5 \
  --train_epochs 10 \
  --patience 3 \
  --batch_size $batch_size \
  --itr 1  | tee logs/ActivityLong7NeuronsTest/TSMixer'_'$seq_len'_'$pred_len.log

# Linear
python -u run_longExp.py \
  --is_training 1 \
  --model Linear \
  --data Activity \
  --root_path $root_path \
  --data_path $data_path \
  --model_id Linear_$seq_len'_'$pred_len \
  --features M \
  --label_len $label_len \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 7 \
  --c_out 7 \
  --loss mae \
  --des 'Exp' \
  --experiment_name $experiment_name \
  --num_workers 5 \
  --train_epochs 10 \
  --patience 3 \
  --batch_size $batch_size \
  --itr 1  | tee logs/ActivityLong7NeuronsTest/Linear'_'$seq_len'_'$pred_len.log

# DLinear
python -u run_longExp.py \
  --is_training 1 \
  --model DLinear \
  --data Activity \
  --root_path $root_path \
  --data_path $data_path \
  --model_id DLinear_$seq_len'_'$pred_len \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 7 \
  --c_out 7 \
  --loss mae \
  --des 'Exp' \
  --experiment_name $experiment_name \
  --num_workers 5 \
  --train_epochs 10 \
  --patience 3 \
  --batch_size $batch_size \
  --itr 1  | tee logs/ActivityLong7NeuronsTest/DLinear'_'$seq_len'_'$pred_len.log

# Informer
python -u run_longExp.py \
  --is_training 1 \
  --model Informer \
  --data Activity \
  --root_path $root_path \
  --data_path $data_path \
  --model_id Informer_$seq_len'_'$pred_len \
  --features M \
  --label_len $label_len \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --loss mae \
  --des 'Exp' \
  --experiment_name $experiment_name \
  --train_epochs 10 \
  --batch_size $batch_size \
  --freq m \
  --itr 1  | tee logs/ActivityLong7NeuronsTest/Informer'_'$seq_len'_'$pred_len.log


# Transformer
python -u run_longExp.py \
  --is_training 1 \
  --model Transformer \
  --data Activity \
  --root_path $root_path \
  --data_path $data_path \
  --model_id Transformer_$seq_len'_'$pred_len \
  --features M \
  --label_len $label_len \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --freq m \
  --loss mae \
  --des 'Exp' \
  --experiment_name $experiment_name \
  --batch_size $batch_size \
  --itr 1  | tee logs/ActivityLong7NeuronsTest/Transformer'_'$seq_len'_'$pred_len.log

# Autoformer
python -u run_longExp.py \
  --is_training 1 \
  --model Autoformer \
  --data Activity \
  --root_path $root_path \
  --data_path $data_path \
  --model_id Autoformer_$seq_len'_'$pred_len \
  --features M \
  --train_epochs 50 \
  --label_len $label_len \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --freq m \
  --loss mae \
  --des 'Exp' \
  --experiment_name $experiment_name \
  --batch_size $batch_size \
  --itr 1  | tee logs/ActivityLong7NeuronsTest/Autoformer'_'$seq_len'_'$pred_len.log

#  sh scripts/EXP_Activity/activity_short.sh
# add --individual for DLinear-I
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/ActivityShort" ]; then
    mkdir ./logs/ActivityShort
fi
seq_len=4
pred_len=1
label_len=4

# # Run Naive baseline (true null model)
python -u run_stat.py \
  --is_training 1 \
  --model Naive \
  --data Activity \
  --root_path ./dataset/ \
  --data_path session_0.h5 \
  --model_id Activity_$seq_len'_'$pred_len \
  --features M \
  --label_len $label_len \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --des 'Exp' \
  --batch_size 128 \
  --itr 1  | tee logs/ActivityShort/Naive'_'$seq_len'_'$pred_len.log

# Run DummyLinear baseline
python -u run_longExp.py \
  --is_training 1 \
  --model DummyLinear \
  --data Activity \
  --root_path ./dataset/ \
  --data_path session_0.h5 \
  --model_id Activity_$seq_len'_'$pred_len \
  --features M \
  --train_epochs 10\
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 5000 \
  --loss mae \
  --des 'Exp' \
  --batch_size 128 \
  --learning_rate 0.01 \
  --itr 1  | tee logs/ActivityShort/DummyLinear'_'$seq_len'_'$pred_len.log

# DLinear
python -u run_longExp.py \
  --is_training 1 \
  --model DLinear \
  --data Activity \
  --root_path ./dataset/ \
  --data_path session_0.h5 \
  --model_id DLinear_$seq_len'_'$pred_len \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 5000 \
  --loss mae \
  --des 'Exp' \
  --num_workers 5 \
  --train_epochs 10 \
  --patience 3 \
  --batch_size 8 \
  --learning_rate 0.01 \
  --itr 1  | tee logs/ActivityShort/DLinear'_'$seq_len'_'$pred_len.log

# Informer
python -u run_longExp.py \
  --is_training 1 \
  --model Informer \
  --data Activity \
  --root_path ./dataset/ \
  --data_path session_0.h5 \
  --model_id Informer_$seq_len'_'$pred_len \
  --features M \
  --label_len $label_len \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 5000 \
  --dec_in 5000 \
  --c_out 5000 \
  --loss mae \
  --des 'Exp' \
  --train_epochs 10 \
  --batch_size 32 \
  --freq m \
  --learning_rate 0.01 \
  --itr 1  | tee logs/ActivityShort/Informer'_'$seq_len'_'$pred_len.log


# Transformer
python -u run_longExp.py \
  --is_training 1 \
  --model Transformer \
  --data Activity \
  --root_path ./dataset/ \
  --data_path session_0.h5 \
  --model_id Transformer_$seq_len'_'$pred_len \
  --features M \
  --label_len $label_len \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 5000 \
  --dec_in 5000 \
  --c_out 5000 \
  --freq m \
  --loss mae \
  --des 'Exp' \
  --batch_size 32 \
  --learning_rate 0.01 \
  --itr 1  | tee logs/ActivityShort/Transformer'_'$seq_len'_'$pred_len.log

# Autoformer
python -u run_longExp.py \
  --is_training 1 \
  --model Autoformer \
  --data Activity \
  --root_path ./dataset/ \
  --data_path session_0.h5 \
  --model_id Autoformer_$seq_len'_'$pred_len \
  --features M \
  --train_epochs 10 \
  --label_len $label_len \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 5000 \
  --dec_in 5000 \
  --c_out 5000 \
  --freq m \
  --loss mae \
  --des 'Exp' \
  --batch_size 32 \
  --learning_rate 0.01 \
  --itr 1  | tee logs/ActivityShort/Autoformer'_'$seq_len'_'$pred_len.log

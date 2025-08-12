#!/bin/bash
# Quick test for TSMixer on Activity dataset
pred_len=8
seq_len=32
label_len=4

echo "Testing TSMixer on Activity dataset..."

python -u run_longExp.py \
  --is_training 1 \
  --model TSMixer \
  --data Activity \
  --root_path ./dataset/ \
  --data_path session_0.h5 \
  --model_id TSMixer_test_$seq_len'_'$pred_len \
  --features M \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len $pred_len \
  --enc_in 5000 \
  --loss mae \
  --des 'Test' \
  --num_workers 1 \
  --train_epochs 1 \
  --patience 3 \
  --batch_size 4 \
  --learning_rate 0.01 \
  --itr 1
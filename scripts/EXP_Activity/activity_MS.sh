
#  sh scripts/EXP_Activity/activity_MS.sh
# add --individual for DLinear-I
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/Activity" ]; then
    mkdir ./logs/Activity
fi
seq_len=336
pred_len=96

# # # Run Naive baseline (true null model)
# python -u run_stat.py \
#   --is_training 1 \
#   --model Naive \
#   --data Activity \
#   --root_path ./dataset/ \
#   --data_path session_0.h5 \
#   --model_id Activity_$seq_len'_'$pred_len \
#   --features M \
#   --seq_len $seq_len \
#   --pred_len $pred_len \
#   --des 'Exp' \
#   --batch_size 128 \
#   --itr 1  | tee logs/Activity/Naive'_'$seq_len'_'$pred_len.log

# # Run DummyLinear baseline
# python -u run_longExp.py \
#   --is_training 1 \
#   --model DummyLinear \
#   --data Activity \
#   --root_path ./dataset/ \
#   --data_path session_0.h5 \
#   --model_id Activity_$seq_len'_'$pred_len \
#   --features M \
#   --train_epochs 1\
#   --seq_len $seq_len \
#   --pred_len $pred_len \
#   --enc_in 5000 \
#   --des 'Exp' \
#   --batch_size 128 \
#   --learning_rate 0.0005 \
#   --itr 1  | tee logs/Activity/DummyLinear'_'$seq_len'_'$pred_len.log

# DLinear
python -u run_longExp.py \
  --is_training 1 \
  --model Autoformer \
  --data ActivityOrdered \
  --root_path ./dataset/ \
  --data_path session_0.h5 \
  --model_id DLinear_MS_$seq_len'_'$pred_len \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 5000 \
  --c_out 5000 \
  --des 'Exp' \
  --dropout 0.1 \
  --batch_size 32 \
  --learning_rate 0.0005 \
  --itr 1  | tee logs/Activity/DLinear_MS'_'$seq_len'_'$pred_len.log

# # Informer
# python -u run_longExp.py \
#   --is_training 1 \
#   --model Informer \
#   --data Activity \
#   --root_path ./dataset/ \
#   --data_path session_0.h5 \
#   --model_id Informer_$seq_len'_'$pred_len \
#   --features M \
#   --seq_len $seq_len \
#   --pred_len $pred_len \
#   --enc_in 5000 \
#   --dec_in 5000 \
#   --c_out 5000 \
#   --des 'Exp' \
#   --dropout 0.1 \
#   --batch_size 64 \
#   --freq m \
#   --learning_rate 0.0005 \
#   --itr 1  | tee logs/Activity/Informer'_'$seq_len'_'$pred_len.log


# # Transformer
# python -u run_longExp.py \
#   --is_training 1 \
#   --model Transformer \
#   --data Activity \
#   --root_path ./dataset/ \
#   --data_path session_0.h5 \
#   --model_id Transformer_$seq_len'_'$pred_len \
#   --features M \
#   --seq_len $seq_len \
#   --pred_len $pred_len \
#   --enc_in 5000 \
#   --dec_in 5000 \
#   --c_out 5000 \
#   --freq m \
#   --des 'Exp' \
#   --batch_size 32 \
#   --learning_rate 0.0005 \
#   --itr 1  | tee logs/Activity/Transformer'_'$seq_len'_'$pred_len.log

# Autoformer
# python -u run_longExp.py \
#   --is_training 1 \
#   --model Autoformer \
#   --data Activity \
#   --root_path ./dataset/ \
#   --data_path session_0.h5 \
#   --model_id Autoformer_MS'_'$seq_len'_'$pred_len \
#   --features MS \
#   --train_epochs 100 \
#   --seq_len $seq_len \
#   --pred_len $pred_len \
#   --enc_in 5000 \
#   --dec_in 5000 \
#   --c_out 10 \
#   --freq m \
#   --des 'Exp' \
#   --batch_size 32 \
#   --learning_rate 0.0005 \
#   --itr 1  | tee logs/Activity/Autoformer'_'MS'_'$seq_len'_'$pred_len.log

#   --dropout 0.1 \
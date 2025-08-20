#  sh scripts/EXP_Activity/activity_short_70_700_7000_neurons.sh
# Experiment to understand the number of neuronal variates and how it effects model performance on a held out test set. 

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

# Arrays for different datasets
experiment_names=("ActivityShort70N" "ActivityShort700N" "ActivityShort7000N")
data_paths=("session_5ea6bb9b-6163-4e8a-816b-efe7002666b0_70.h5" "session_5ea6bb9b-6163-4e8a-816b-efe7002666b0_700.h5" "session_5ea6bb9b-6163-4e8a-816b-efe7002666b0_7000.h5")
neuron_counts=(70 700 7000)

batch_size=8
pred_len=8
seq_len=16
label_len=8
root_path=/cs/student/projects1/aibh/2024/gcosta/mpci_data/

# Loop through each dataset
for i in {0..2}; do
    experiment_name=${experiment_names[$i]}
    data_path=${data_paths[$i]}
    enc_in=${neuron_counts[$i]}
    c_out=${neuron_counts[$i]}
    
    echo "Running experiments for $experiment_name with $enc_in neurons..."
    
    if [ ! -d "./logs/$experiment_name" ]; then
        mkdir ./logs/$experiment_name
    fi


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
  --experiment_name $experiment_name \
  --batch_size $batch_size \
  --itr 1  | tee logs/$experiment_name/Naive'_'$seq_len'_'$pred_len.log

# Mean baseline
python -u run_stat.py \
  --is_training 1 \
  --model Mean \
  --data Activity \
  --root_path $root_path \
  --data_path $data_path \
  --model_id Mean_$seq_len'_'$pred_len \
  --features M \
  --label_len $label_len \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --des 'Exp' \
  --experiment_name $experiment_name \
  --batch_size $batch_size \
  --itr 1  | tee logs/$experiment_name/Mean'_'$seq_len'_'$pred_len.log


# POCO
python -u run_longExp.py \
  --is_training 1 \
  --model POCO \
  --data Activity \
  --root_path $root_path \
  --data_path $data_path \
  --model_id POCO_$seq_len'_'$pred_len \
  --features M \
  --label_len $label_len \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in $enc_in \
  --c_out $c_out \
  --loss mae \
  --des 'Exp' \
  --experiment_name $experiment_name \
  --num_workers 5 \
  --train_epochs 10 \
  --patience 3 \
  --batch_size $batch_size \
  --itr 1  | tee logs/$experiment_name/POCO'_'$seq_len'_'$pred_len.log

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
  --enc_in $enc_in \
  --c_out $c_out \
  --loss mae \
  --des 'Exp' \
  --experiment_name $experiment_name \
  --num_workers 5 \
  --train_epochs 10 \
  --patience 3 \
  --batch_size $batch_size \
  --itr 1  | tee logs/$experiment_name/TSMixer'_'$seq_len'_'$pred_len.log

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
  --enc_in $enc_in \
  --c_out $c_out \
  --loss mae \
  --des 'Exp' \
  --experiment_name $experiment_name \
  --num_workers 5 \
  --train_epochs 10 \
  --patience 3 \
  --batch_size $batch_size \
  --itr 1  | tee logs/$experiment_name/Linear'_'$seq_len'_'$pred_len.log

# DLinear
python -u run_longExp.py \
  --is_training 1 \
  --model DLinear \
  --data Activity \
  --root_path $root_path \
  --data_path $data_path \
  --model_id DLinear_$seq_len'_'$pred_len \
  --features M \
  --label_len $label_len \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in $enc_in \
  --c_out $c_out \
  --loss mae \
  --des 'Exp' \
  --experiment_name $experiment_name \
  --num_workers 5 \
  --train_epochs 10 \
  --patience 3 \
  --batch_size $batch_size \
  --itr 1  | tee logs/$experiment_name/DLinear'_'$seq_len'_'$pred_len.log

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
  --enc_in $enc_in \
  --dec_in $enc_in \
  --c_out $c_out \
  --loss mae \
  --des 'Exp' \
  --experiment_name $experiment_name \
  --train_epochs 10 \
  --batch_size $batch_size \
  --freq m \
  --itr 1  | tee logs/$experiment_name/Informer'_'$seq_len'_'$pred_len.log


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
  --enc_in $enc_in \
  --dec_in $enc_in \
  --c_out $c_out \
  --freq m \
  --loss mae \
  --des 'Exp' \
  --experiment_name $experiment_name \
  --batch_size $batch_size \
  --itr 1  | tee logs/$experiment_name/Transformer'_'$seq_len'_'$pred_len.log

done
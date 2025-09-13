#  sh scripts/EXP_Activity_Behavioral/activity_long_active_11392_neurons_parallel.sh
# Behavorial split and how it effects model performance on unseen behavioral data. In this case, just active data

# Signal handler for Ctrl+C
cleanup() {
    echo "Caught interrupt signal. Killing all background jobs..."
    jobs -p | xargs -r kill
    exit 1
}
trap cleanup INT TERM

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

# Single dataset configuration
experiment_name="ActivityLongActive11392N"
data_path="session_5ea6bb9b-6163-4e8a-816b-efe7002666b0_11392.h5"
neuron_count=11392
enc_in=$neuron_count
c_out=$neuron_count

batch_size=8
pred_len=16
seq_len=48
label_len=16
root_path=/cs/student/projects1/aibh/2024/gcosta/mpci_data/

# Create log directory for experiment
if [ ! -d "./logs/$experiment_name" ]; then
    mkdir ./logs/$experiment_name
fi

echo "Starting parallel training experiments..."

# Pair 1: Run Naive and Mean models in parallel
echo "Running Pair 1: Naive and Mean models..."

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
  --itr 1  | tee logs/$experiment_name/Naive'_'$seq_len'_'$pred_len.log &

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
  --itr 1  | tee logs/$experiment_name/Mean'_'$seq_len'_'$pred_len.log &

wait
echo "Pair 1 completed: Naive and Mean models."

# Pair 2: Run TSMixer and POCO models in parallel
echo "Running Pair 2: TSMixer and POCO models..."

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
  --feature_idx 69 \
  --itr 1  | tee logs/$experiment_name/TSMixer'_'$seq_len'_'$pred_len.log &

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
  --patience 1 \
  --batch_size $batch_size \
  --feature_idx 69 \
  --itr 1  | tee logs/$experiment_name/POCO'_'$seq_len'_'$pred_len.log &

wait
echo "Pair 2 completed: TSMixer and POCO models."

# Pair 3: Run Linear and DLinear models in parallel
echo "Running Pair 3: Linear and DLinear models..."

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
  --feature_idx 69 \
  --itr 1  | tee logs/$experiment_name/Linear'_'$seq_len'_'$pred_len.log &

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
  --feature_idx 69 \
  --itr 1  | tee logs/$experiment_name/DLinear'_'$seq_len'_'$pred_len.log &

wait
echo "Pair 3 completed: Linear and DLinear models."

# Pair 4: Run Informer and Transformer models in parallel
echo "Running Pair 4: Informer and Transformer models..."

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
  --feature_idx 69 \
  --itr 1  | tee logs/$experiment_name/Informer'_'$seq_len'_'$pred_len.log &

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
  --feature_idx 69 \
  --itr 1  | tee logs/$experiment_name/Transformer'_'$seq_len'_'$pred_len.log &

wait
echo "Pair 4 completed: Informer and Transformer models."

echo "All experiments completed!"

# gcosta@mandarin-l ~/mpci-transformer$ cd utils && python output_session.py --session_id 5ea6bb9b-6163-4e8a-816b-efe7002666b0 --num_neurons 11392
# 3.10.14 (main, May  6 2024, 19:42:50) [GCC 11.2.0]
# Preprocessing session
# Loading session from ONE
# ================================================================================
# CALCIUM IMAGING PREPROCESSING PIPELINE
# ================================================================================

# 1. Loading ΔF/F data...
# MEAN: 13.615189552307129
#    Raw ΔF/F shape: (22074, 11393)
#    Raw ΔF/F range: -4594.4 to 19492.6

# 1.5. Truncating passive video period...
#    Last stimOff at 3667.2s, truncating at index 18309
#    Truncated shape: (18309, 11393)
#    Duration: 4513.4s → 3665.1s (81.2% retained)

# 2. Applying robust normalization...
#    Normalized range: -0.250 to 1.500
#    Original mean: 13.403, std: 143.550
#    Normalized mean: 0.341, std: 0.198

# 3. Selecting first 11392 neurons...
# Selected neurons mean: 0.34100010991096497
#    Selected first 11392 neurons from 11393 total neurons
#    Selection method: first N neurons

# 5. Calculating aligned wheel velocity...
#    Wheel velocity range: -2.500 to 2.500 rad/s
#    Non-zero velocity samples: 18190/18309

# 6. Building covariate matrix...
#    Covariate matrix shape: (18309, 11)
#    Features: 11 total
#    Non-zero covariate samples: [18190  3347   316   359   672   666  1675  2096  2113  2493 13737]

# 7. Quality assessment...

# 6. Saving processed data to /cs/student/projects1/aibh/2024/gcosta/mpci_data/session_5ea6bb9b-6163-4e8a-816b-efe7002666b0_11392.h5
#    Saved (18309, 11392) activity matrix
#    Saved (18309, 11) covariate matrix
#    File size: 712.50 MB
# Preprocessed Session 5ea6bb9b-6163-4e8a-816b-efe7002666b0
# gcosta@mandarin-l utils$ 
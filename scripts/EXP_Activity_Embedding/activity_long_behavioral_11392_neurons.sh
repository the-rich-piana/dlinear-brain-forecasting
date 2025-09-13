#  sh scripts/EXP_Activity_Embedding/activity_long_behavioral_11392_neurons.sh
# Behavorial split and how it effects model performance on unseen behavioral data.

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
experiment_name="ActivityEmbedding"
data_path="session_5ea6bb9b-6163-4e8a-816b-efe7002666b0_11392_passive.h5"
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
# Pair 2: Run TSMixer and POCO models in parallel
echo "Running Pair 1 POCO models..."

python -u run_longExp.py \
  --is_training 1 \
  --model POCO \
  --data ActivityBehavioral \
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


echo "All experiments completed!"
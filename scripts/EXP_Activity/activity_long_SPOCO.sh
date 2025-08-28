#  sh scripts/EXP_Activity/activity_long_SPOCO.sh
# Parallel experiment to understand the number of neuronal variates and how it effects model performance on a held out test set. 

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

# Arrays for different datasets
experiment_names=("ActivityLongSPOCO7000N")
data_paths=("session_5ea6bb9b-6163-4e8a-816b-efe7002666b0_7000.h5")
neuron_counts=(7000)

batch_size=8
pred_len=16
seq_len=48
label_len=16
root_path=/cs/student/projects1/aibh/2024/gcosta/mpci_data/

# Create log directories for all experiments
for i in {0..1}; do
    experiment_name=${experiment_names[$i]}
    if [ ! -d "./logs/$experiment_name" ]; then
        mkdir ./logs/$experiment_name
    fi
done

echo "Starting parallel training experiments..."

# Run POCO for all datasets in parallel
echo "Running SPOCO models..."
for i in {0..2}; do
    experiment_name=${experiment_names[$i]}
    data_path=${data_paths[$i]}
    enc_in=${neuron_counts[$i]}
    c_out=${neuron_counts[$i]}
    
    python -u run_longExp.py \
      --is_training 1 \
      --model SPOCO \
      --data ActivityStimulus \
      --root_path $root_path \
      --data_path $data_path \
      --model_id SPOCO_$seq_len'_'$pred_len \
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
      --itr 1  | tee logs/$experiment_name/SPOCO'_'$seq_len'_'$pred_len.log &
done
wait
echo "SPOCO models completed."

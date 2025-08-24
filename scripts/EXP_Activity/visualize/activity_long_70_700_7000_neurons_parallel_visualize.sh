#  sh scripts/EXP_Activity/visualize/activity_long_70_700_7000_neurons_parallel_visualize.sh
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
experiment_names=("ActivityLong70N" "ActivityLong700N" "ActivityLong7000N")
data_paths=("session_5ea6bb9b-6163-4e8a-816b-efe7002666b0_70.h5" "session_5ea6bb9b-6163-4e8a-816b-efe7002666b0_700.h5" "session_5ea6bb9b-6163-4e8a-816b-efe7002666b0_7000.h5")
neuron_counts=(70 700 7000)

batch_size=8
pred_len=16
seq_len=48
label_len=16
root_path=/cs/student/projects1/aibh/2024/gcosta/mpci_data/

# Create log directories for all experiments
for i in {0..2}; do
    experiment_name=${experiment_names[$i]}
    if [ ! -d "./logs/$experiment_name" ]; then
        mkdir ./logs/$experiment_name
    fi
done

echo "Starting visualizing and testing of experiments..."

# Run TSMixer for all datasets in parallel
echo "Running TSMixer models..."
for i in {0..2}; do
    experiment_name=${experiment_names[$i]}
    data_path=${data_paths[$i]}
    enc_in=${neuron_counts[$i]}
    c_out=${neuron_counts[$i]}
    
    python -u run_longExp.py \
      --is_training 0 \
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
done
wait
echo "TSMixer models completed."

# Run POCO for all datasets in parallel
echo "Running POCO models..."
for i in {0..2}; do
    experiment_name=${experiment_names[$i]}
    data_path=${data_paths[$i]}
    enc_in=${neuron_counts[$i]}
    c_out=${neuron_counts[$i]}
    
    python -u run_longExp.py \
      --is_training 0 \
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
done
wait
echo "POCO models completed."

# Run Linear for all datasets in parallel
echo "Running Linear models..."
for i in {0..2}; do
    experiment_name=${experiment_names[$i]}
    data_path=${data_paths[$i]}
    enc_in=${neuron_counts[$i]}
    c_out=${neuron_counts[$i]}
    
    python -u run_longExp.py \
      --is_training 0 \
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
done
wait
echo "Linear models completed."

# Run DLinear for all datasets in parallel
echo "Running DLinear models..."
for i in {0..2}; do
    experiment_name=${experiment_names[$i]}
    data_path=${data_paths[$i]}
    enc_in=${neuron_counts[$i]}
    c_out=${neuron_counts[$i]}
    
    python -u run_longExp.py \
      --is_training 0 \
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
done
wait
echo "DLinear models completed."

# Run Informer for all datasets in parallel
echo "Running Informer models..."
for i in {0..2}; do
    experiment_name=${experiment_names[$i]}
    data_path=${data_paths[$i]}
    enc_in=${neuron_counts[$i]}
    c_out=${neuron_counts[$i]}
    
    python -u run_longExp.py \
      --is_training 0 \
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
done
wait
echo "Informer models completed."

# Run Transformer for all datasets in parallel
echo "Running Transformer models..."
for i in {0..2}; do
    experiment_name=${experiment_names[$i]}
    data_path=${data_paths[$i]}
    enc_in=${neuron_counts[$i]}
    c_out=${neuron_counts[$i]}
    
    python -u run_longExp.py \
      --is_training 0 \
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
done
wait
echo "Transformer models completed."

echo "All experiments completed!"
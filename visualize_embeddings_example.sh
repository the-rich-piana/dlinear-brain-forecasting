#!/bin/bash
# Example script to visualize POCO embeddings from your trained model

# Set paths
CHECKPOINT_PATH="./checkpoints/ActivityLongBehavioral7000N/POCO_48_16_POCO_ActivityBehavioral_ftM_sl48_ll16_pl16_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_Exp_0/checkpoint.pth"
OUTPUT_DIR="./embedding_visualizations"

# Create output directory
mkdir -p $OUTPUT_DIR

echo "Visualizing POCO embeddings..."
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Output directory: $OUTPUT_DIR"

# Run the visualization script
python visualize_poco_embeddings.py \
    --checkpoint $CHECKPOINT_PATH \
    --num_neurons 7000 \
    --methods PCA UMAP TSNE \
    --output_dir $OUTPUT_DIR

echo "Done! Check $OUTPUT_DIR for visualization results."
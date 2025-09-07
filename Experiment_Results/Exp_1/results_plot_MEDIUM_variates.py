import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.calculate_performance_metrics import calculate_performance_metrics

sns.set_theme(style="whitegrid")
plt.rcParams.update({'figure.dpi': 300, 'savefig.dpi': 300})

resultsMedium70 = pd.read_csv("ActivityLong70N.csv")
resultsMedium700 = pd.read_csv("ActivityLong700N.csv") 
resultsMedium7000 = pd.read_csv("ActivityLong7000N.csv")

medium_metrics_70 = calculate_performance_metrics(resultsMedium70, "Medium_70N")
medium_metrics_700 = calculate_performance_metrics(resultsMedium700, "Medium_700N")
medium_metrics_7000 = calculate_performance_metrics(resultsMedium7000, "Medium_7000N")

combined_metrics = pd.concat([
    medium_metrics_70.assign(neurons=70),
    medium_metrics_700.assign(neurons=700),
    medium_metrics_7000.assign(neurons=7000)
], ignore_index=True)

models_to_plot = ['Mean', 'POCO', 'TSMixer', 'Linear', 'DLinear', 'Informer', 'Transformer']
plot_data = combined_metrics[combined_metrics['model'].isin(models_to_plot)]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

colors = sns.color_palette("viridis", n_colors=3)

ax1 = sns.barplot(data=plot_data, x='model', y='mae', hue='neurons', 
                  palette=colors, ax=axes[0])
axes[0].set_title('Raw MAE Performance', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Model', fontsize=12)
axes[0].set_ylabel('Mean Absolute Error', fontsize=12)
axes[0].tick_params(axis='x', rotation=45)
axes[0].legend(title='Neurons', bbox_to_anchor=(1.05, 1), loc='upper left')
axes[0].grid(axis='y', alpha=0.3)

ax2 = sns.barplot(data=plot_data, x='model', y='mae_improvement_pct', hue='neurons',
                  palette=colors, ax=axes[1])
axes[1].set_title('MAE Improvement over Naive (%)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Model', fontsize=12)
axes[1].set_ylabel('Improvement Percentage', fontsize=12)
axes[1].tick_params(axis='x', rotation=45)
axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
axes[1].legend(title='Neurons', bbox_to_anchor=(1.05, 1), loc='upper left')
axes[1].grid(axis='y', alpha=0.3)

ax3 = sns.barplot(data=plot_data, x='model', y='prediction_score', hue='neurons',
                  palette=colors, ax=axes[2])
axes[2].set_title('Prediction Score', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Model', fontsize=12)
axes[2].set_ylabel('Score (1 - L(model)/L(naive))', fontsize=12)
axes[2].tick_params(axis='x', rotation=45)
axes[2].axhline(y=0, color='red', linestyle='--', alpha=0.7)
axes[2].legend(title='Neurons', bbox_to_anchor=(1.05, 1), loc='upper left')
axes[2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('Experiment_Results/figures/medium_performance_comparison.png', 
            bbox_inches='tight', dpi=300)
plt.show()

improvement_pivot = plot_data.pivot_table(
    values='mae_improvement_pct', 
    index='model', 
    columns='neurons', 
    fill_value=0
)

plt.figure(figsize=(8, 6))
ax = sns.heatmap(improvement_pivot, annot=True, cmap='RdYlGn', center=0,
                 fmt='.1f', cbar_kws={'label': 'MAE Improvement (%)'})
plt.title('Performance Improvement Heatmap\n(MAE % Improvement over Naive)', 
          fontsize=14, fontweight='bold')
plt.xlabel('Number of Neurons', fontsize=12)
plt.ylabel('Model', fontsize=12)
plt.tight_layout()
plt.savefig('Experiment_Results/figures/medium_improvement_heatmap.png', 
            bbox_inches='tight', dpi=300)
plt.show()

plt.figure(figsize=(12, 8))
for model in models_to_plot:
    model_data = plot_data[plot_data['model'] == model]
    plt.plot(model_data['neurons'], model_data['mae_improvement_pct'], 
             marker='o', linewidth=2, markersize=8, label=model)

plt.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Naive Baseline')
plt.xlabel('Number of Neurons', fontsize=12, fontweight='bold')
plt.ylabel('MAE Improvement over Naive (%)', fontsize=12, fontweight='bold')
plt.title('Model Scaling Efficiency\n(Performance vs. Number of Neuronal Variates)', 
          fontsize=14, fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.xscale('log')
plt.xticks([70, 700, 7000], ['70', '700', '7000'])
plt.tight_layout()
plt.savefig('Experiment_Results/figures/medium_scaling_efficiency.png', 
            bbox_inches='tight', dpi=300)
plt.show()

print("\nSummary: Model Performance Improvement (%)")
print("=" * 50)
summary_table = improvement_pivot.round(1)
print(summary_table)

print("\nBest performing models by neuron count:")
print("=" * 40)
for neurons in [70, 700, 7000]:
    best_model = improvement_pivot[neurons].idxmax()
    best_score = improvement_pivot[neurons].max()
    print(f"{neurons:4} neurons: {best_model:12} (+{best_score:5.1f}%)")

print("\nModels benefiting most from additional variates:")
print("=" * 50)
scaling_benefit = improvement_pivot[7000] - improvement_pivot[70]
scaling_sorted = scaling_benefit.sort_values(ascending=False)
for model, benefit in scaling_sorted.items():
    print(f"{model:12}: {improvement_pivot.loc[model, 70]:+6.1f}% → "
          f"{improvement_pivot.loc[model, 7000]:+6.1f}% (Δ{benefit:+5.1f}%)")
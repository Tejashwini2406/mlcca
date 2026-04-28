import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

PASTEL_PALETTE = ['#FFB3D9', '#A5D8FF', '#D0B3FF', '#B2F2BB']

def _get_model_palette(models):
    """Map each model name to a consistent color from the pastel palette. 🎀"""
    return {model: PASTEL_PALETTE[i % len(PASTEL_PALETTE)] for i, model in enumerate(sorted(models))}

def calculate_stability_metrics(df):
    """Groups raw metric vault by model to calculate mean and std of F1-scores. 🎀"""
    stability_df = df.groupby('Model').agg({
        'F1_Score': ['mean', 'std', 'min', 'max'],
        'Train_Acc': ['mean', 'std'],
        'Test_Acc': ['mean', 'std'],
        'Generalization_Gap': ['mean', 'std']
    }).reset_index()

    stability_df.columns = ['Model', 'F1_mean', 'F1_std', 'F1_min', 'F1_max',
                             'Train_Acc_mean', 'Train_Acc_std',
                             'Test_Acc_mean', 'Test_Acc_std',
                             'Gen_Gap_mean', 'Gen_Gap_std']

    stability_df['Stability_Index'] = stability_df['F1_mean'] * (1 - stability_df['F1_std'])

    return stability_df.sort_values(by='Stability_Index', ascending=False)

def generate_stability_dashboard(raw_vault, output_path='stability_dashboard.png'):
    """Generate comprehensive visualization dashboard. 🎀"""
    fig = plt.figure(figsize=(16, 12))

    model_palette = _get_model_palette(raw_vault['Model'].unique())

    ax1 = plt.subplot(3, 3, 1)
    sns.violinplot(x='Model', y='F1_Score', data=raw_vault, inner="quartile", hue='Model',
                   palette=model_palette, legend=False)
    ax1.set_title("F1-Score Distribution (Violin Plot)", fontsize=12, fontweight='bold')
    ax1.set_ylabel("F1-Score")
    ax1.tick_params(axis='x', rotation=30)

    ax2 = plt.subplot(3, 3, 2)
    sns.boxplot(x='Model', y='Test_Acc', data=raw_vault, hue='Model',
                palette=model_palette, legend=False)
    ax2.set_title("Test Accuracy Distribution (Box Plot)", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Test Accuracy")
    ax2.tick_params(axis='x', rotation=30)

    ax3 = plt.subplot(3, 3, 3)
    summary = raw_vault.groupby('Model')['F1_Score'].agg(['mean', 'std']).reset_index()
    for idx, row in summary.iterrows():
        color = model_palette.get(row['Model'], '#A5D8FF')
        ax3.scatter(row['mean'], row['std'], s=300, alpha=0.85, label=row['Model'], color=color, edgecolors='black', linewidth=1)
        ax3.annotate(row['Model'], (row['mean'], row['std']), fontsize=9, ha='center', fontweight='bold')
    ax3.set_xlabel("Mean F1-Score", fontweight='bold')
    ax3.set_ylabel("Std Dev F1-Score", fontweight='bold')
    ax3.set_title("The Sweet Spot: Mean vs. Variance", fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(frameon=True, shadow=True, loc='best')

    ax4 = plt.subplot(3, 3, 4)
    sns.boxplot(x='Model', y='Generalization_Gap', data=raw_vault, hue='Model',
                palette=model_palette, legend=False)
    ax4.set_title("Generalization Gap (Train-Test)", fontsize=12, fontweight='bold')
    ax4.set_ylabel("Generalization Gap")
    ax4.axhline(y=0, color='#FFB3D9', linestyle='--', alpha=0.7, linewidth=2)
    ax4.tick_params(axis='x', rotation=30)

    ax5 = plt.subplot(3, 3, 5)
    train_means = raw_vault.groupby('Model')['Train_Acc'].mean()
    test_means = raw_vault.groupby('Model')['Test_Acc'].mean()
    x = np.arange(len(train_means))
    width = 0.35
    bars1 = ax5.bar(x - width/2, train_means, width, label='Train Accuracy', color='#A5D8FF', alpha=0.9, edgecolor='black', linewidth=0.5)
    bars2 = ax5.bar(x + width/2, test_means, width, label='Test Accuracy', color='#FFB3D9', alpha=0.9, edgecolor='black', linewidth=0.5)
    ax5.set_ylabel("Accuracy", fontweight='bold')
    ax5.set_title("Train vs Test Accuracy", fontsize=12, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(train_means.index, rotation=30, ha='right')
    ax5.legend(frameon=True, shadow=True)
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.set_ylim(0, 1.05)
    for bar in bars1:
        height = bar.get_height()
        ax5.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=7, fontweight='bold')
    for bar in bars2:
        height = bar.get_height()
        ax5.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax6 = plt.subplot(3, 3, 6)
    stability = raw_vault.groupby('Model')['F1_Score'].agg(lambda x: x.mean() * (1 - x.std())).sort_values(ascending=True)
    colors = [model_palette.get(m, '#A5D8FF') for m in stability.index]
    ax6.barh(stability.index, stability.values, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax6.set_xlabel("Stability Index (SI = μ × (1-σ))", fontweight='bold')
    ax6.set_title("Model Stability Index Ranking", fontsize=12, fontweight='bold')
    for i, (model, v) in enumerate(stability.items()):
        ax6.text(v + 0.01, i, f'{v:.4f}', va='center', fontweight='bold', color=model_palette.get(model, 'black'))

    ax7 = plt.subplot(3, 3, 7)
    for model in sorted(raw_vault['Model'].unique()):
        model_data = raw_vault[raw_vault['Model'] == model]['F1_Score']
        color = model_palette.get(model, '#A5D8FF')
        ax7.hist(model_data, alpha=0.6, label=model, bins=15, color=color, edgecolor='black', linewidth=0.3)
    ax7.set_xlabel("F1-Score", fontweight='bold')
    ax7.set_ylabel("Frequency", fontweight='bold')
    ax7.legend(frameon=True, shadow=True, fontsize=8, loc='upper left', bbox_to_anchor=(1.02, 1))
    ax7.grid(True, alpha=0.3, axis='y')

    ax8 = plt.subplot(3, 3, 8)
    line_styles = ['-', '--', '-.', ':']
    models_sorted = sorted(raw_vault['Model'].unique())
    for i, model in enumerate(models_sorted):
        model_data = raw_vault[raw_vault['Model'] == model].reset_index(drop=True)
        cumulative_mean = model_data['F1_Score'].expanding().mean()
        color = model_palette.get(model, '#A5D8FF')
        ax8.plot(cumulative_mean.index, cumulative_mean, label=model, linewidth=2.5,
                 linestyle=line_styles[i % len(line_styles)], alpha=0.85, color=color)
    ax8.set_xlabel("Simulation Run", fontweight='bold')
    ax8.set_ylabel("Cumulative Mean F1-Score", fontweight='bold')
    ax8.set_title("Convergence Analysis: Cumulative F1-Score Mean", fontsize=12, fontweight='bold')
    ax8.legend(frameon=True, shadow=True, fontsize=8, loc='upper left', bbox_to_anchor=(1.02, 1))
    ax8.grid(True, alpha=0.3)

    ax9 = plt.subplot(3, 3, 9)
    stats_data = raw_vault.groupby('Model')[['F1_Score', 'Train_Acc', 'Test_Acc']].mean()
    from matplotlib.colors import LinearSegmentedColormap
    pastel_cmap = LinearSegmentedColormap.from_list("pastel", ["#FFB3D9", "#A5D8FF", "#B2F2BB"])
    sns.heatmap(stats_data.T, annot=True, fmt='.3f', cmap=pastel_cmap, ax=ax9, cbar_kws={'label': 'Score'}, linewidths=0.5)
    ax9.set_title("Mean Performance Heatmap", fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n[*] Dashboard saved as '{output_path}'")
    plt.close(fig)


"""
Week 7 Loss Curves and Training Dynamics Visualization
Creates loss curves, accuracy curves, and other line graphs from training results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Set style for publication-quality plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 10

# Define data paths
DATA_DIR = Path(__file__).parent
OUTPUT_DIR = DATA_DIR / 'visualizations_loss_curves'
OUTPUT_DIR.mkdir(exist_ok=True)

def load_hyperparameter_results():
    """Load hyperparameter tuning results"""
    csv_file = DATA_DIR / 'hyperparameter_tuning_results.csv'
    if csv_file.exists():
        return pd.read_csv(csv_file)
    return None

def create_loss_curves_by_learning_rate():
    """Create loss curves comparison across different learning rates"""
    df = load_hyperparameter_results()
    if df is None:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Week 7: Training vs Validation Loss by Learning Rate', 
                fontsize=14, fontweight='bold')
    
    learning_rates = df['learning_rate'].unique()
    learning_rates = sorted(learning_rates)
    
    for idx, lr in enumerate(learning_rates):
        ax = axes[idx]
        lr_data = df[df['learning_rate'] == lr].sort_values('batch_size')
        
        # Plot training loss
        ax.plot(range(len(lr_data)), lr_data['final_train_loss'], 
               marker='o', linewidth=2.5, markersize=10, label='Training Loss', color='#3498db')
        
        # Plot validation loss
        ax.plot(range(len(lr_data)), lr_data['final_val_loss'], 
               marker='s', linewidth=2.5, markersize=10, label='Validation Loss', color='#e74c3c')
        
        ax.set_xlabel('Batch Size Configuration', fontweight='bold')
        ax.set_ylabel('Loss', fontweight='bold')
        ax.set_title(f'Learning Rate = {lr}', fontweight='bold')
        ax.set_xticks(range(len(lr_data)))
        ax.set_xticklabels([f'BS={int(bs)}' for bs in lr_data['batch_size']], rotation=45)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'week7_loss_by_learning_rate.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: week7_loss_by_learning_rate.png")
    plt.close()

def create_loss_curves_by_batch_size():
    """Create loss curves comparison across different batch sizes"""
    df = load_hyperparameter_results()
    if df is None:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Week 7: Training vs Validation Loss by Batch Size', 
                fontsize=14, fontweight='bold')
    
    batch_sizes = sorted(df['batch_size'].unique())
    
    for idx, bs in enumerate(batch_sizes):
        ax = axes[idx]
        bs_data = df[df['batch_size'] == bs].sort_values('learning_rate')
        
        # Plot training loss
        ax.plot(range(len(bs_data)), bs_data['final_train_loss'], 
               marker='o', linewidth=2.5, markersize=10, label='Training Loss', color='#2ecc71')
        
        # Plot validation loss
        ax.plot(range(len(bs_data)), bs_data['final_val_loss'], 
               marker='s', linewidth=2.5, markersize=10, label='Validation Loss', color='#f39c12')
        
        ax.set_xlabel('Learning Rate Configuration', fontweight='bold')
        ax.set_ylabel('Loss', fontweight='bold')
        ax.set_title(f'Batch Size = {int(bs)}', fontweight='bold')
        ax.set_xticks(range(len(bs_data)))
        ax.set_xticklabels([f'LR={lr}' for lr in bs_data['learning_rate']], rotation=45)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'week7_loss_by_batch_size.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: week7_loss_by_batch_size.png")
    plt.close()

def create_training_accuracy_curves():
    """Create training accuracy curves"""
    df = load_hyperparameter_results()
    if df is None:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Week 7: Training and Validation Accuracy Trends', fontsize=14, fontweight='bold')
    
    # Plot 1: Train Acc by Learning Rate
    ax = axes[0, 0]
    for lr in sorted(df['learning_rate'].unique()):
        lr_data = df[df['learning_rate'] == lr].sort_values('batch_size')
        ax.plot(range(len(lr_data)), lr_data['final_train_acc'], 
               marker='o', linewidth=2, label=f'LR={lr}', markersize=8)
    ax.set_xlabel('Batch Size', fontweight='bold')
    ax.set_ylabel('Accuracy', fontweight='bold')
    ax.set_title('Training Accuracy by Learning Rate', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Val Acc by Learning Rate
    ax = axes[0, 1]
    for lr in sorted(df['learning_rate'].unique()):
        lr_data = df[df['learning_rate'] == lr].sort_values('batch_size')
        ax.plot(range(len(lr_data)), lr_data['final_val_acc'], 
               marker='s', linewidth=2, label=f'LR={lr}', markersize=8)
    ax.set_xlabel('Batch Size', fontweight='bold')
    ax.set_ylabel('Accuracy', fontweight='bold')
    ax.set_title('Validation Accuracy by Learning Rate', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Train Acc by Batch Size
    ax = axes[1, 0]
    for bs in sorted(df['batch_size'].unique()):
        bs_data = df[df['batch_size'] == bs].sort_values('learning_rate')
        ax.plot(range(len(bs_data)), bs_data['final_train_acc'], 
               marker='o', linewidth=2, label=f'BS={int(bs)}', markersize=8)
    ax.set_xlabel('Learning Rate', fontweight='bold')
    ax.set_ylabel('Accuracy', fontweight='bold')
    ax.set_title('Training Accuracy by Batch Size', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Val Acc by Batch Size
    ax = axes[1, 1]
    for bs in sorted(df['batch_size'].unique()):
        bs_data = df[df['batch_size'] == bs].sort_values('learning_rate')
        ax.plot(range(len(bs_data)), bs_data['final_val_acc'], 
               marker='s', linewidth=2, label=f'BS={int(bs)}', markersize=8)
    ax.set_xlabel('Learning Rate', fontweight='bold')
    ax.set_ylabel('Accuracy', fontweight='bold')
    ax.set_title('Validation Accuracy by Batch Size', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'week7_accuracy_curves.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: week7_accuracy_curves.png")
    plt.close()

def create_loss_gap_analysis():
    """Analyze and visualize overfitting/underfitting through loss gap"""
    df = load_hyperparameter_results()
    if df is None:
        return
    
    df['loss_gap'] = df['final_val_loss'] - df['final_train_loss']
    df['acc_gap'] = df['final_train_acc'] - df['final_val_acc']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Week 7: Overfitting Analysis - Loss and Accuracy Gaps', 
                fontsize=14, fontweight='bold')
    
    # Loss Gap
    ax = axes[0]
    colors = ['#2ecc71' if gap < 1 else '#f39c12' if gap < 3 else '#e74c3c' 
              for gap in df['loss_gap']]
    bars = ax.bar(range(len(df)), df['loss_gap'], color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Configuration', fontweight='bold')
    ax.set_ylabel('Loss Gap (Val - Train)', fontweight='bold')
    ax.set_title('Loss Gap (Overfitting Indicator)', fontweight='bold')
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([f'LR={row["learning_rate"]}\nBS={int(row["batch_size"])}' 
                        for _, row in df.iterrows()], fontsize=8, rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, (bar, gap) in enumerate(zip(bars, df['loss_gap'])):
        ax.text(bar.get_x() + bar.get_width()/2, gap + 0.2, f'{gap:.2f}', 
               ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Accuracy Gap
    ax = axes[1]
    colors = ['#2ecc71' if gap < 0.05 else '#f39c12' if gap < 0.15 else '#e74c3c' 
              for gap in df['acc_gap']]
    bars = ax.bar(range(len(df)), df['acc_gap'], color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Configuration', fontweight='bold')
    ax.set_ylabel('Accuracy Gap (Train - Val)', fontweight='bold')
    ax.set_title('Accuracy Gap (Overfitting Indicator)', fontweight='bold')
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([f'LR={row["learning_rate"]}\nBS={int(row["batch_size"])}' 
                        for _, row in df.iterrows()], fontsize=8, rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, (bar, gap) in enumerate(zip(bars, df['acc_gap'])):
        ax.text(bar.get_x() + bar.get_width()/2, gap + 0.005, f'{gap:.4f}', 
               ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'week7_loss_gap_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: week7_loss_gap_analysis.png")
    plt.close()

def create_test_metrics_progression():
    """Create line graphs showing test metrics progression"""
    df = load_hyperparameter_results()
    if df is None:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Week 7: Test Metrics Progression Across Configurations', 
                fontsize=14, fontweight='bold')
    
    # Test Accuracy
    ax = axes[0, 0]
    ax.plot(range(len(df)), df['test_accuracy'], marker='o', linewidth=2.5, 
           markersize=10, color='#3498db', label='Test Accuracy')
    ax.fill_between(range(len(df)), df['test_accuracy'], alpha=0.3, color='#3498db')
    ax.set_xlabel('Configuration', fontweight='bold')
    ax.set_ylabel('Accuracy', fontweight='bold')
    ax.set_title('Test Accuracy Progression', fontweight='bold')
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([f'{i}' for i in range(len(df))])
    ax.grid(True, alpha=0.3)
    
    # Test F1-Score
    ax = axes[0, 1]
    ax.plot(range(len(df)), df['test_f1'], marker='s', linewidth=2.5, 
           markersize=10, color='#e74c3c', label='Test F1-Score')
    ax.fill_between(range(len(df)), df['test_f1'], alpha=0.3, color='#e74c3c')
    ax.set_xlabel('Configuration', fontweight='bold')
    ax.set_ylabel('F1-Score', fontweight='bold')
    ax.set_title('Test F1-Score Progression', fontweight='bold')
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([f'{i}' for i in range(len(df))])
    ax.grid(True, alpha=0.3)
    
    # Precision
    ax = axes[1, 0]
    ax.plot(range(len(df)), df['precision'], marker='^', linewidth=2.5, 
           markersize=10, color='#2ecc71', label='Precision')
    ax.fill_between(range(len(df)), df['precision'], alpha=0.3, color='#2ecc71')
    ax.set_xlabel('Configuration', fontweight='bold')
    ax.set_ylabel('Precision', fontweight='bold')
    ax.set_title('Precision Progression', fontweight='bold')
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([f'{i}' for i in range(len(df))])
    ax.grid(True, alpha=0.3)
    
    # Recall
    ax = axes[1, 1]
    ax.plot(range(len(df)), df['recall'], marker='D', linewidth=2.5, 
           markersize=10, color='#f39c12', label='Recall')
    ax.fill_between(range(len(df)), df['recall'], alpha=0.3, color='#f39c12')
    ax.set_xlabel('Configuration', fontweight='bold')
    ax.set_ylabel('Recall', fontweight='bold')
    ax.set_title('Recall Progression', fontweight='bold')
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([f'{i}' for i in range(len(df))])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'week7_test_metrics_progression.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: week7_test_metrics_progression.png")
    plt.close()

def create_training_time_vs_performance():
    """Create visualization of training time vs performance metrics"""
    df = load_hyperparameter_results()
    if df is None:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Week 7: Training Time vs Performance Trade-off', 
                fontsize=14, fontweight='bold')
    
    # Training Time vs Accuracy
    ax = axes[0]
    scatter = ax.scatter(df['training_time']/3600, df['test_accuracy'], 
                        s=300, c=df['learning_rate'], cmap='viridis', 
                        alpha=0.7, edgecolors='black', linewidth=1.5)
    ax.set_xlabel('Training Time (hours)', fontweight='bold')
    ax.set_ylabel('Test Accuracy', fontweight='bold')
    ax.set_title('Training Time vs Test Accuracy', fontweight='bold')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Learning Rate', fontweight='bold')
    
    # Add annotations
    for i, (time, acc) in enumerate(zip(df['training_time']/3600, df['test_accuracy'])):
        ax.annotate(f'{i}', (time, acc), fontsize=9, fontweight='bold', 
                   ha='center', va='center')
    
    # Training Time vs F1-Score
    ax = axes[1]
    scatter = ax.scatter(df['training_time']/3600, df['test_f1'], 
                        s=300, c=df['batch_size'], cmap='plasma', 
                        alpha=0.7, edgecolors='black', linewidth=1.5)
    ax.set_xlabel('Training Time (hours)', fontweight='bold')
    ax.set_ylabel('Test F1-Score', fontweight='bold')
    ax.set_title('Training Time vs Test F1-Score', fontweight='bold')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Batch Size', fontweight='bold')
    
    # Add annotations
    for i, (time, f1) in enumerate(zip(df['training_time']/3600, df['test_f1'])):
        ax.annotate(f'{i}', (time, f1), fontsize=9, fontweight='bold', 
                   ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'week7_training_time_vs_performance.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: week7_training_time_vs_performance.png")
    plt.close()

def create_comprehensive_loss_curve():
    """Create a comprehensive loss curve comparison for all configurations"""
    df = load_hyperparameter_results()
    if df is None:
        return
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    configs = []
    for _, row in df.iterrows():
        config = f"LR={row['learning_rate']}\nBS={int(row['batch_size'])}"
        configs.append(config)
    
    x_pos = range(len(df))
    width = 0.2
    
    # Create grouped bars
    bars1 = ax.bar([i - 1.5*width for i in x_pos], df['final_train_loss'], 
                   width, label='Training Loss', color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar([i - 0.5*width for i in x_pos], df['final_val_loss'], 
                   width, label='Validation Loss', color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Configuration', fontweight='bold', fontsize=12)
    ax.set_ylabel('Loss', fontweight='bold', fontsize=12)
    ax.set_title('Week 7: Complete Loss Comparison Across All Configurations', 
                fontweight='bold', fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(configs, fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'week7_comprehensive_loss_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: week7_comprehensive_loss_comparison.png")
    plt.close()

def create_epochs_trained_analysis():
    """Analyze relationship between epochs trained and performance"""
    df = load_hyperparameter_results()
    if df is None:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Week 7: Epochs Trained vs Performance Metrics', fontsize=14, fontweight='bold')
    
    # Epochs vs Test Accuracy
    ax = axes[0, 0]
    ax.plot(df['epochs_trained'], df['test_accuracy'], marker='o', linewidth=2.5, 
           markersize=10, color='#3498db', label='Test Accuracy')
    ax.scatter(df['epochs_trained'], df['test_accuracy'], s=200, c='#3498db', 
              alpha=0.6, edgecolors='black', linewidth=1.5)
    ax.set_xlabel('Epochs Trained', fontweight='bold')
    ax.set_ylabel('Test Accuracy', fontweight='bold')
    ax.set_title('Epochs Trained vs Test Accuracy', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Epochs vs Test F1
    ax = axes[0, 1]
    ax.plot(df['epochs_trained'], df['test_f1'], marker='s', linewidth=2.5, 
           markersize=10, color='#e74c3c', label='Test F1-Score')
    ax.scatter(df['epochs_trained'], df['test_f1'], s=200, c='#e74c3c', 
              alpha=0.6, edgecolors='black', linewidth=1.5)
    ax.set_xlabel('Epochs Trained', fontweight='bold')
    ax.set_ylabel('Test F1-Score', fontweight='bold')
    ax.set_title('Epochs Trained vs Test F1-Score', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Epochs vs Final Val Loss
    ax = axes[1, 0]
    ax.plot(df['epochs_trained'], df['final_val_loss'], marker='^', linewidth=2.5, 
           markersize=10, color='#2ecc71', label='Val Loss')
    ax.scatter(df['epochs_trained'], df['final_val_loss'], s=200, c='#2ecc71', 
              alpha=0.6, edgecolors='black', linewidth=1.5)
    ax.set_xlabel('Epochs Trained', fontweight='bold')
    ax.set_ylabel('Validation Loss', fontweight='bold')
    ax.set_title('Epochs Trained vs Validation Loss', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Training Time vs Epochs (Efficiency)
    ax = axes[1, 1]
    ax.plot(df['epochs_trained'], df['training_time']/3600, marker='D', linewidth=2.5, 
           markersize=10, color='#f39c12', label='Training Time')
    ax.scatter(df['epochs_trained'], df['training_time']/3600, s=200, c='#f39c12', 
              alpha=0.6, edgecolors='black', linewidth=1.5)
    ax.set_xlabel('Epochs Trained', fontweight='bold')
    ax.set_ylabel('Training Time (hours)', fontweight='bold')
    ax.set_title('Epochs Trained vs Training Time', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'week7_epochs_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: week7_epochs_analysis.png")
    plt.close()

def create_multi_metric_line_plot():
    """Create a multi-metric line plot showing all performance metrics"""
    df = load_hyperparameter_results()
    if df is None:
        return
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Normalize all metrics to 0-1 scale for comparison
    x_pos = range(len(df))
    
    ax.plot(x_pos, df['test_accuracy'], marker='o', linewidth=2.5, markersize=8, 
           label='Test Accuracy', color='#3498db')
    ax.plot(x_pos, df['test_f1'], marker='s', linewidth=2.5, markersize=8, 
           label='Test F1-Score', color='#e74c3c')
    ax.plot(x_pos, df['precision'], marker='^', linewidth=2.5, markersize=8, 
           label='Precision', color='#2ecc71')
    ax.plot(x_pos, df['recall'], marker='D', linewidth=2.5, markersize=8, 
           label='Recall', color='#f39c12')
    
    # Add validation accuracy line
    ax.plot(x_pos, df['val_accuracy'], marker='v', linewidth=2.5, markersize=8, 
           label='Validation Accuracy', color='#9b59b6', linestyle='--')
    
    ax.set_xlabel('Configuration Index', fontweight='bold', fontsize=12)
    ax.set_ylabel('Score', fontweight='bold', fontsize=12)
    ax.set_title('Week 7: Multi-Metric Performance Comparison Across Configurations', 
                fontweight='bold', fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{i}' for i in x_pos])
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'week7_multi_metric_progression.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: week7_multi_metric_progression.png")
    plt.close()

def generate_loss_curve_report():
    """Generate text report of loss curve analysis"""
    df = load_hyperparameter_results()
    if df is None:
        return
    
    report = []
    report.append("=" * 90)
    report.append("WEEK 7 LOSS CURVES AND TRAINING DYNAMICS - DETAILED ANALYSIS REPORT")
    report.append("=" * 90)
    report.append("")
    
    # Overall statistics
    report.append("LOSS STATISTICS")
    report.append("-" * 90)
    report.append(f"Training Loss - Mean: {df['final_train_loss'].mean():.6f}, "
                 f"Std: {df['final_train_loss'].std():.6f}")
    report.append(f"Validation Loss - Mean: {df['final_val_loss'].mean():.6f}, "
                 f"Std: {df['final_val_loss'].std():.6f}")
    report.append(f"Loss Gap (Val-Train) - Mean: {(df['final_val_loss'] - df['final_train_loss']).mean():.6f}")
    report.append("")
    
    # Best and worst configurations
    best_idx = df['test_accuracy'].idxmax()
    worst_idx = df['test_accuracy'].idxmin()
    
    report.append("BEST CONFIGURATION")
    report.append("-" * 90)
    best_row = df.loc[best_idx]
    report.append(f"Learning Rate: {best_row['learning_rate']}")
    report.append(f"Batch Size: {int(best_row['batch_size'])}")
    report.append(f"Training Loss: {best_row['final_train_loss']:.6f}")
    report.append(f"Validation Loss: {best_row['final_val_loss']:.6f}")
    report.append(f"Loss Gap: {best_row['final_val_loss'] - best_row['final_train_loss']:.6f}")
    report.append(f"Test Accuracy: {best_row['test_accuracy']:.6f}")
    report.append(f"Test F1-Score: {best_row['test_f1']:.6f}")
    report.append(f"Epochs Trained: {int(best_row['epochs_trained'])}")
    report.append(f"Training Time: {best_row['training_time']/3600:.2f} hours")
    report.append("")
    
    report.append("WORST CONFIGURATION")
    report.append("-" * 90)
    worst_row = df.loc[worst_idx]
    report.append(f"Learning Rate: {worst_row['learning_rate']}")
    report.append(f"Batch Size: {int(worst_row['batch_size'])}")
    report.append(f"Training Loss: {worst_row['final_train_loss']:.6f}")
    report.append(f"Validation Loss: {worst_row['final_val_loss']:.6f}")
    report.append(f"Loss Gap: {worst_row['final_val_loss'] - worst_row['final_train_loss']:.6f}")
    report.append(f"Test Accuracy: {worst_row['test_accuracy']:.6f}")
    report.append(f"Test F1-Score: {worst_row['test_f1']:.6f}")
    report.append("")
    
    # Overfitting analysis
    report.append("OVERFITTING ANALYSIS")
    report.append("-" * 90)
    df['loss_gap'] = df['final_val_loss'] - df['final_train_loss']
    
    low_gap = df[df['loss_gap'] < 1]
    med_gap = df[(df['loss_gap'] >= 1) & (df['loss_gap'] < 3)]
    high_gap = df[df['loss_gap'] >= 3]
    
    report.append(f"Low Overfitting (Gap < 1): {len(low_gap)} configurations")
    report.append(f"Medium Overfitting (1 <= Gap < 3): {len(med_gap)} configurations")
    report.append(f"High Overfitting (Gap >= 3): {len(high_gap)} configurations")
    report.append("")
    
    # Training efficiency
    report.append("TRAINING EFFICIENCY")
    report.append("-" * 90)
    df['time_per_epoch'] = df['training_time'] / df['epochs_trained']
    report.append(f"Average Time per Epoch: {df['time_per_epoch'].mean():.2f} seconds")
    report.append(f"Fastest Configuration: LR={df.loc[df['time_per_epoch'].idxmin(), 'learning_rate']}, "
                 f"BS={int(df.loc[df['time_per_epoch'].idxmin(), 'batch_size'])} "
                 f"({df['time_per_epoch'].min():.2f}s/epoch)")
    report.append(f"Slowest Configuration: LR={df.loc[df['time_per_epoch'].idxmax(), 'learning_rate']}, "
                 f"BS={int(df.loc[df['time_per_epoch'].idxmax(), 'batch_size'])} "
                 f"({df['time_per_epoch'].max():.2f}s/epoch)")
    report.append("")
    
    report.append("=" * 90)
    
    report_text = "\n".join(report)
    report_file = OUTPUT_DIR / 'week7_loss_curves_report.txt'
    with open(report_file, 'w') as f:
        f.write(report_text)
    
    print("✓ Saved: week7_loss_curves_report.txt")
    return report_text

def main():
    """Main execution function"""
    print("\n" + "=" * 90)
    print("WEEK 7 LOSS CURVES AND TRAINING DYNAMICS VISUALIZATION")
    print("=" * 90 + "\n")
    
    df = load_hyperparameter_results()
    if df is None:
        print("❌ Hyperparameter tuning results file not found!")
        return
    
    print(f"✓ Loaded {len(df)} training configurations\n")
    print("Generating loss curves and line graphs...")
    
    # Generate all visualizations
    create_loss_curves_by_learning_rate()
    create_loss_curves_by_batch_size()
    create_training_accuracy_curves()
    create_loss_gap_analysis()
    create_test_metrics_progression()
    create_training_time_vs_performance()
    create_comprehensive_loss_curve()
    create_epochs_trained_analysis()
    create_multi_metric_line_plot()
    
    print("\nGenerating detailed analysis report...")
    report = generate_loss_curve_report()
    
    print(f"\n✓ All visualizations saved to: {OUTPUT_DIR}")
    print("\n" + "=" * 90)
    print("ANALYSIS COMPLETE!")
    print("=" * 90 + "\n")
    
    print(report)

if __name__ == "__main__":
    main()

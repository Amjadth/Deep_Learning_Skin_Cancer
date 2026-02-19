"""
Week 9 Loss Curves and Training Dynamics Visualization
Creates comprehensive loss curves and training visualizations for EfficientNetB0 and EfficientNetB3
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
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / 'visualizations_week9_loss_curves'
OUTPUT_DIR.mkdir(exist_ok=True)

# Model info
MODELS = {
    'EfficientNetB0': {
        'color': '#3498db',
        'path': BASE_DIR / 'EfficientNetB0'
    },
    'EfficientNetB3': {
        'color': '#e74c3c',
        'path': BASE_DIR / 'EfficientNetB3'
    }
}

def load_training_history(model_name):
    """Load training history for a model (phase1 + phase2)"""
    model_path = MODELS[model_name]['path']
    
    # Load phase 1
    phase1_file = None
    phase2_file = None
    
    for file in model_path.glob('phase1*'):
        phase1_file = file
        break
    
    for file in model_path.glob('phase2*'):
        phase2_file = file
        break
    
    if phase1_file is None:
        return None, None
    
    phase1_df = pd.read_csv(phase1_file)
    phase1_df['phase'] = 1
    
    if phase2_file is not None:
        phase2_df = pd.read_csv(phase2_file)
        phase2_df['phase'] = 2
        # Adjust epoch numbers for phase 2
        max_epoch = phase1_df['epoch'].max()
        phase2_df['epoch'] = phase2_df['epoch'] + max_epoch + 1
        history_df = pd.concat([phase1_df, phase2_df], ignore_index=True)
    else:
        history_df = phase1_df
    
    return history_df, phase1_df

def create_individual_loss_curves():
    """Create loss curves for each model"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Week 9: Training vs Validation Loss - Individual Models', 
                fontsize=14, fontweight='bold')
    
    for idx, (model_name, model_info) in enumerate(MODELS.items()):
        ax = axes[idx]
        history, _ = load_training_history(model_name)
        
        if history is None:
            continue
        
        ax.plot(history['epoch'], history['loss'], 
               marker='o', linewidth=2, markersize=4, label='Training Loss', 
               color=model_info['color'], alpha=0.7)
        ax.plot(history['epoch'], history['val_loss'], 
               marker='s', linewidth=2, markersize=4, label='Validation Loss', 
               color=model_info['color'], alpha=0.4, linestyle='--')
        
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Loss', fontweight='bold')
        ax.set_title(f'{model_name} - Loss Curves', fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'week9_individual_loss_curves.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: week9_individual_loss_curves.png")
    plt.close()

def create_combined_loss_comparison():
    """Create combined loss curves for both models"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Week 9: Model Comparison - Training and Validation Loss', 
                fontsize=14, fontweight='bold')
    
    # Training Loss
    ax = axes[0]
    for model_name, model_info in MODELS.items():
        history, _ = load_training_history(model_name)
        if history is not None:
            ax.plot(history['epoch'], history['loss'], 
                   linewidth=2.5, label=model_name, color=model_info['color'], marker='o', markersize=3)
    
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Training Loss', fontweight='bold')
    ax.set_title('Training Loss Comparison', fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Validation Loss
    ax = axes[1]
    for model_name, model_info in MODELS.items():
        history, _ = load_training_history(model_name)
        if history is not None:
            ax.plot(history['epoch'], history['val_loss'], 
                   linewidth=2.5, label=model_name, color=model_info['color'], 
                   marker='s', markersize=3, linestyle='--')
    
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Validation Loss', fontweight='bold')
    ax.set_title('Validation Loss Comparison', fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'week9_combined_loss_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: week9_combined_loss_comparison.png")
    plt.close()

def create_accuracy_curves():
    """Create accuracy curves for both models"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Week 9: Training vs Validation Accuracy', 
                fontsize=14, fontweight='bold')
    
    # Individual model comparison
    for idx, (model_name, model_info) in enumerate(MODELS.items()):
        ax = axes[idx]
        history, _ = load_training_history(model_name)
        
        if history is None:
            continue
        
        ax.plot(history['epoch'], history['accuracy'], 
               marker='o', linewidth=2, markersize=4, label='Training Accuracy', 
               color=model_info['color'], alpha=0.7)
        ax.plot(history['epoch'], history['val_accuracy'], 
               marker='s', linewidth=2, markersize=4, label='Validation Accuracy', 
               color=model_info['color'], alpha=0.4, linestyle='--')
        
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Accuracy', fontweight='bold')
        ax.set_title(f'{model_name} - Accuracy Curves', fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'week9_accuracy_curves.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: week9_accuracy_curves.png")
    plt.close()

def create_combined_accuracy_comparison():
    """Create combined accuracy comparison"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Week 9: Model Comparison - Training and Validation Accuracy', 
                fontsize=14, fontweight='bold')
    
    # Training Accuracy
    ax = axes[0]
    for model_name, model_info in MODELS.items():
        history, _ = load_training_history(model_name)
        if history is not None:
            ax.plot(history['epoch'], history['accuracy'], 
                   linewidth=2.5, label=model_name, color=model_info['color'], marker='o', markersize=3)
    
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Training Accuracy', fontweight='bold')
    ax.set_title('Training Accuracy Comparison', fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Validation Accuracy
    ax = axes[1]
    for model_name, model_info in MODELS.items():
        history, _ = load_training_history(model_name)
        if history is not None:
            ax.plot(history['epoch'], history['val_accuracy'], 
                   linewidth=2.5, label=model_name, color=model_info['color'], 
                   marker='s', markersize=3, linestyle='--')
    
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Validation Accuracy', fontweight='bold')
    ax.set_title('Validation Accuracy Comparison', fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'week9_combined_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: week9_combined_accuracy_comparison.png")
    plt.close()

def create_learning_rate_schedule():
    """Visualize learning rate schedule changes"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Week 9: Learning Rate Schedule Throughout Training', 
                fontsize=14, fontweight='bold')
    
    for idx, (model_name, model_info) in enumerate(MODELS.items()):
        ax = axes[idx]
        history, _ = load_training_history(model_name)
        
        if history is None or 'lr' not in history.columns:
            continue
        
        # Plot learning rate
        ax.plot(history['epoch'], history['lr'], 
               linewidth=3, color=model_info['color'], marker='o', markersize=5)
        
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Learning Rate (log scale)', fontweight='bold')
        ax.set_title(f'{model_name} - Learning Rate Schedule', fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'week9_learning_rate_schedule.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: week9_learning_rate_schedule.png")
    plt.close()

def create_overfitting_analysis():
    """Analyze overfitting through loss and accuracy gaps"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Week 9: Overfitting Analysis', fontsize=14, fontweight='bold')
    
    for idx, (model_name, model_info) in enumerate(MODELS.items()):
        history, _ = load_training_history(model_name)
        if history is None:
            continue
        
        # Loss gap
        ax = axes[idx, 0]
        loss_gap = history['val_loss'] - history['loss']
        ax.fill_between(history['epoch'], loss_gap, alpha=0.3, color=model_info['color'])
        ax.plot(history['epoch'], loss_gap, linewidth=2.5, color=model_info['color'], marker='o', markersize=4)
        ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Loss Gap (Val - Train)', fontweight='bold')
        ax.set_title(f'{model_name} - Loss Gap Analysis', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Accuracy gap
        ax = axes[idx, 1]
        acc_gap = history['accuracy'] - history['val_accuracy']
        ax.fill_between(history['epoch'], acc_gap, alpha=0.3, color=model_info['color'])
        ax.plot(history['epoch'], acc_gap, linewidth=2.5, color=model_info['color'], marker='s', markersize=4)
        ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Accuracy Gap (Train - Val)', fontweight='bold')
        ax.set_title(f'{model_name} - Accuracy Gap Analysis', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'week9_overfitting_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: week9_overfitting_analysis.png")
    plt.close()

def create_training_phases_comparison():
    """Compare phase 1 and phase 2 training"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Week 9: Training Phases Comparison', fontsize=14, fontweight='bold')
    
    for idx, (model_name, model_info) in enumerate(MODELS.items()):
        history, phase1 = load_training_history(model_name)
        if history is None:
            continue
        
        phase2 = history[history['phase'] == 2]
        
        # Loss comparison
        ax = axes[idx, 0]
        ax.plot(phase1['epoch'], phase1['loss'], linewidth=2, label='Phase 1 - Train', 
               color=model_info['color'], marker='o', markersize=3)
        ax.plot(phase1['epoch'], phase1['val_loss'], linewidth=2, label='Phase 1 - Val', 
               color=model_info['color'], marker='o', markersize=3, linestyle='--', alpha=0.6)
        
        if len(phase2) > 0:
            ax.plot(phase2['epoch'], phase2['loss'], linewidth=2, label='Phase 2 - Train', 
                   color=model_info['color'], marker='s', markersize=3, alpha=0.7)
            ax.plot(phase2['epoch'], phase2['val_loss'], linewidth=2, label='Phase 2 - Val', 
                   color=model_info['color'], marker='s', markersize=3, linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Loss', fontweight='bold')
        ax.set_title(f'{model_name} - Loss by Phase', fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Accuracy comparison
        ax = axes[idx, 1]
        ax.plot(phase1['epoch'], phase1['accuracy'], linewidth=2, label='Phase 1 - Train', 
               color=model_info['color'], marker='o', markersize=3)
        ax.plot(phase1['epoch'], phase1['val_accuracy'], linewidth=2, label='Phase 1 - Val', 
               color=model_info['color'], marker='o', markersize=3, linestyle='--', alpha=0.6)
        
        if len(phase2) > 0:
            ax.plot(phase2['epoch'], phase2['accuracy'], linewidth=2, label='Phase 2 - Train', 
                   color=model_info['color'], marker='s', markersize=3, alpha=0.7)
            ax.plot(phase2['epoch'], phase2['val_accuracy'], linewidth=2, label='Phase 2 - Val', 
                   color=model_info['color'], marker='s', markersize=3, linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Accuracy', fontweight='bold')
        ax.set_title(f'{model_name} - Accuracy by Phase', fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'week9_training_phases.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: week9_training_phases.png")
    plt.close()

def create_convergence_speed_comparison():
    """Compare convergence speed of both models"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Week 9: Convergence Speed Comparison', fontsize=14, fontweight='bold')
    
    # Target accuracy for convergence analysis
    target_accuracies = [0.6, 0.65, 0.70]
    
    convergence_data = {}
    for model_name, model_info in MODELS.items():
        history, _ = load_training_history(model_name)
        if history is None:
            continue
        
        convergence_data[model_name] = []
        for target_acc in target_accuracies:
            epoch_reached = None
            for _, row in history.iterrows():
                if row['val_accuracy'] >= target_acc:
                    epoch_reached = row['epoch']
                    break
            convergence_data[model_name].append(epoch_reached if epoch_reached is not None else -1)
    
    # Plot convergence speed
    ax = axes[0]
    x_pos = np.arange(len(target_accuracies))
    width = 0.35
    
    for i, (model_name, model_info) in enumerate(MODELS.items()):
        if model_name in convergence_data:
            bars = ax.bar(x_pos + i*width, convergence_data[model_name], width, 
                         label=model_name, color=model_info['color'], alpha=0.8, edgecolor='black', linewidth=1.5)
            
            # Add value labels
            for bar, val in zip(bars, convergence_data[model_name]):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, val + 1, f'{int(val)}', 
                           ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    ax.set_ylabel('Epoch Number', fontweight='bold')
    ax.set_title('Epochs to Reach Target Validation Accuracy', fontweight='bold')
    ax.set_xticks(x_pos + width/2)
    ax.set_xticklabels([f'{int(a*100)}%' for a in target_accuracies])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Final metrics comparison
    ax = axes[1]
    final_metrics = []
    labels = []
    
    for model_name, model_info in MODELS.items():
        history, _ = load_training_history(model_name)
        if history is not None:
            final_row = history.iloc[-1]
            final_metrics.append([
                final_row['accuracy'],
                final_row['val_accuracy'],
                final_row['loss'],
                final_row['val_loss']
            ])
            labels.append(model_name)
    
    # Plot final validation accuracy
    x_pos = np.arange(len(labels))
    val_accs = [m[1] for m in final_metrics]
    train_accs = [m[0] for m in final_metrics]
    
    ax.bar(x_pos - 0.2, train_accs, 0.4, label='Final Training Accuracy', 
          color='#95a5a6', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.bar(x_pos + 0.2, val_accs, 0.4, label='Final Validation Accuracy', 
          color='#2c3e50', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Accuracy', fontweight='bold')
    ax.set_title('Final Training Metrics Comparison', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (ta, va) in enumerate(zip(train_accs, val_accs)):
        ax.text(i - 0.2, ta + 0.02, f'{ta:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        ax.text(i + 0.2, va + 0.02, f'{va:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'week9_convergence_speed.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: week9_convergence_speed.png")
    plt.close()

def create_comprehensive_metrics_plot():
    """Create comprehensive plot with all metrics"""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Week 9: Comprehensive Training Metrics Comparison', fontsize=16, fontweight='bold')
    
    for idx, (model_name, model_info) in enumerate(MODELS.items()):
        history, _ = load_training_history(model_name)
        if history is None:
            continue
        
        col = idx
        
        # Training Loss
        ax = fig.add_subplot(gs[0, col])
        ax.plot(history['epoch'], history['loss'], linewidth=2.5, color=model_info['color'], marker='o', markersize=3)
        ax.fill_between(history['epoch'], history['loss'], alpha=0.2, color=model_info['color'])
        ax.set_ylabel('Training Loss', fontweight='bold')
        ax.set_title(f'{model_name} - Training Loss', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Validation Loss
        ax = fig.add_subplot(gs[1, col])
        ax.plot(history['epoch'], history['val_loss'], linewidth=2.5, color=model_info['color'], marker='s', markersize=3)
        ax.fill_between(history['epoch'], history['val_loss'], alpha=0.2, color=model_info['color'])
        ax.set_ylabel('Validation Loss', fontweight='bold')
        ax.set_title(f'{model_name} - Validation Loss', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Validation Accuracy
        ax = fig.add_subplot(gs[2, col])
        ax.plot(history['epoch'], history['val_accuracy'], linewidth=2.5, color=model_info['color'], marker='^', markersize=3)
        ax.fill_between(history['epoch'], history['val_accuracy'], alpha=0.2, color=model_info['color'])
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Validation Accuracy', fontweight='bold')
        ax.set_title(f'{model_name} - Validation Accuracy', fontweight='bold')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
    
    plt.savefig(OUTPUT_DIR / 'week9_comprehensive_metrics.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: week9_comprehensive_metrics.png")
    plt.close()

def generate_loss_curve_report():
    """Generate detailed analysis report"""
    report = []
    report.append("=" * 100)
    report.append("WEEK 9 LOSS CURVES AND TRAINING DYNAMICS - COMPREHENSIVE ANALYSIS REPORT")
    report.append("=" * 100)
    report.append("")
    
    for model_name, model_info in MODELS.items():
        history, phase1 = load_training_history(model_name)
        if history is None:
            continue
        
        phase2 = history[history['phase'] == 2]
        
        report.append(f"\n{model_name.upper()}")
        report.append("-" * 100)
        
        # Overall statistics
        report.append("\nOVERALL STATISTICS:")
        report.append(f"  Total Epochs: {int(history['epoch'].max()) + 1}")
        report.append(f"  Phase 1 Epochs: {int(phase1['epoch'].max()) + 1}")
        if len(phase2) > 0:
            report.append(f"  Phase 2 Epochs: {len(phase2)}")
        report.append("")
        
        # Loss analysis
        report.append("LOSS ANALYSIS:")
        report.append(f"  Initial Training Loss: {history['loss'].iloc[0]:.6f}")
        report.append(f"  Final Training Loss: {history['loss'].iloc[-1]:.6f}")
        report.append(f"  Min Training Loss: {history['loss'].min():.6f}")
        report.append(f"  Initial Validation Loss: {history['val_loss'].iloc[0]:.6f}")
        report.append(f"  Final Validation Loss: {history['val_loss'].iloc[-1]:.6f}")
        report.append(f"  Min Validation Loss: {history['val_loss'].min():.6f}")
        report.append("")
        
        # Accuracy analysis
        report.append("ACCURACY ANALYSIS:")
        report.append(f"  Initial Training Accuracy: {history['accuracy'].iloc[0]:.6f}")
        report.append(f"  Final Training Accuracy: {history['accuracy'].iloc[-1]:.6f}")
        report.append(f"  Initial Validation Accuracy: {history['val_accuracy'].iloc[0]:.6f}")
        report.append(f"  Final Validation Accuracy: {history['val_accuracy'].iloc[-1]:.6f}")
        report.append(f"  Max Validation Accuracy: {history['val_accuracy'].max():.6f}")
        report.append("")
        
        # Overfitting analysis
        report.append("OVERFITTING ANALYSIS:")
        final_loss_gap = history['val_loss'].iloc[-1] - history['loss'].iloc[-1]
        final_acc_gap = history['accuracy'].iloc[-1] - history['val_accuracy'].iloc[-1]
        avg_loss_gap = (history['val_loss'] - history['loss']).mean()
        
        report.append(f"  Final Loss Gap (Val - Train): {final_loss_gap:.6f}")
        report.append(f"  Final Accuracy Gap (Train - Val): {final_acc_gap:.6f}")
        report.append(f"  Average Loss Gap: {avg_loss_gap:.6f}")
        
        if final_loss_gap < 0.5:
            report.append("  Status: Minimal overfitting - Good generalization")
        elif final_loss_gap < 1.0:
            report.append("  Status: Moderate overfitting - Acceptable")
        else:
            report.append("  Status: Significant overfitting")
        report.append("")
        
        # Learning rate schedule
        if 'lr' in history.columns:
            report.append("LEARNING RATE SCHEDULE:")
            unique_lrs = history[['epoch', 'lr']].drop_duplicates(subset=['lr'], keep='first')
            for _, row in unique_lrs.iterrows():
                report.append(f"  Epoch {int(row['epoch'])}: LR = {row['lr']}")
            report.append("")
    
    report.append("=" * 100)
    
    report_text = "\n".join(report)
    report_file = OUTPUT_DIR / 'week9_loss_curves_report.txt'
    with open(report_file, 'w') as f:
        f.write(report_text)
    
    print("✓ Saved: week9_loss_curves_report.txt")
    return report_text

def main():
    """Main execution function"""
    print("\n" + "=" * 100)
    print("WEEK 9 LOSS CURVES AND TRAINING DYNAMICS VISUALIZATION")
    print("=" * 100 + "\n")
    
    # Check if data exists
    models_found = 0
    for model_name, model_info in MODELS.items():
        history, _ = load_training_history(model_name)
        if history is not None:
            models_found += 1
            total_epochs = int(history['epoch'].max()) + 1
            print(f"✓ {model_name}: {total_epochs} epochs loaded")
    
    if models_found == 0:
        print("❌ No training history found!")
        return
    
    print(f"\n✓ Found {models_found} Week 9 models\n")
    print("Generating loss curves and training visualizations...")
    
    # Generate all visualizations
    create_individual_loss_curves()
    create_combined_loss_comparison()
    create_accuracy_curves()
    create_combined_accuracy_comparison()
    create_learning_rate_schedule()
    create_overfitting_analysis()
    create_training_phases_comparison()
    create_convergence_speed_comparison()
    create_comprehensive_metrics_plot()
    
    print("\nGenerating detailed analysis report...")
    report = generate_loss_curve_report()
    
    print(f"\n✓ All visualizations saved to: {OUTPUT_DIR}")
    print("\n" + "=" * 100)
    print("ANALYSIS COMPLETE!")
    print("=" * 100 + "\n")
    
    print(report)

if __name__ == "__main__":
    main()

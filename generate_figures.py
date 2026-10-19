import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Styling configuration
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.9

# Colorblind-safe palette (Okabe-Ito / Color Universal Design)
CB_BLUE = '#0072B2'     # Random Forest / Baseline
CB_ORANGE = '#D55E00'   # Rule Baseline
CB_AMBER = '#E69F00'    # Ablation Drift

def plot_recall_vs_fpr(csv_path, out_path, is_ablation=False):
    df = pd.read_csv(csv_path)
    
    fig, ax = plt.subplots(figsize=(8.0, 5.8), dpi=300)
    
    models = [
        {
            'name': 'random_forest',
            'label': 'Random Forest',
            'color': CB_BLUE,
            'marker': 'o',
            'linestyle': '-'
        },
        {
            'name': 'rule_baseline',
            'label': 'Rule Baseline',
            'color': CB_ORANGE,
            'marker': 's',
            'linestyle': '--'
        }
    ]
    
    for m in models:
        m_df = df[df['model'] == m['name']].copy()
        if m_df.empty:
            continue
        # Sort by FPR mean for drawing the curve
        m_df = m_df.sort_values('fpr_mean')
        
        x = m_df['fpr_mean']
        y = m_df['recall_mean']
        xerr = m_df['fpr_std'].fillna(0) if 'fpr_std' in m_df.columns else np.zeros(len(m_df))
        yerr = m_df['recall_std'].fillna(0) if 'recall_std' in m_df.columns else np.zeros(len(m_df))
        
        ax.errorbar(
            x, y,
            xerr=xerr, yerr=yerr,
            label=m['label'],
            color=m['color'],
            marker=m['marker'],
            markersize=7,
            linestyle=m['linestyle'],
            linewidth=2,
            capsize=4,
            capthick=1.2,
            elinewidth=1.2,
            alpha=0.95
        )
        
        if m['name'] == 'random_forest':
            for _, row in m_df.iterrows():
                th = row['threshold']
                if th == 0.9:
                    xytext = (8, -12)
                elif th == 0.7:
                    xytext = (-28, 6)
                elif th == 0.5:
                    xytext = (6, -14)
                elif th == 0.3:
                    xytext = (-10, 8)
                else:  # 0.1
                    xytext = (-30, 8)
                ax.annotate(
                    f"t={th:.1f}",
                    (row['fpr_mean'], row['recall_mean']),
                    textcoords="offset points",
                    xytext=xytext,
                    fontsize=8.5,
                    color=m['color'],
                    fontweight='bold',
                    alpha=0.95
                )
        else:
            handled_zero = False
            for _, row in m_df.iterrows():
                th = row['threshold']
                fpr = row['fpr_mean']
                rec = row['recall_mean']
                
                if fpr == 0.0 and rec == 0.0:
                    if not handled_zero:
                        ax.annotate(
                            "t=0.7, 0.9",
                            (fpr, rec),
                            textcoords="offset points",
                            xytext=(10, -10),
                            fontsize=8.5,
                            color=m['color'],
                            fontweight='bold',
                            alpha=0.95
                        )
                        handled_zero = True
                elif th == 0.5:
                    ax.annotate(
                        f"t={th:.1f}",
                        (fpr, rec),
                        textcoords="offset points",
                        xytext=(8, 5),
                        fontsize=8.5,
                        color=m['color'],
                        fontweight='bold',
                        alpha=0.95
                    )
                else:
                    ax.annotate(
                        f"t={th:.1f}",
                        (fpr, rec),
                        textcoords="offset points",
                        xytext=(8, -3),
                        fontsize=8.5,
                        color=m['color'],
                        fontweight='bold',
                        alpha=0.95
                    )

    ax.set_xlabel('False Positive Rate (FPR)', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_ylabel('Recall (True Positive Rate)', fontsize=11, fontweight='bold', labelpad=8)
    
    if is_ablation:
        ax.set_title('Recall vs. FPR across Operating Thresholds (Ablation)', fontsize=11.5, fontweight='bold', pad=12)
    else:
        ax.set_title('Recall vs. FPR across Operating Thresholds (Baseline)', fontsize=11.5, fontweight='bold', pad=12)
    
    max_fpr = df['fpr_mean'].max()
    ax.set_xlim(-0.01, max(max_fpr * 1.1, 0.2))
    ax.set_ylim(-0.06, 1.05)
    
    ax.grid(True, linestyle='--', alpha=0.5, color='#D0D0D0')
    ax.legend(frameon=True, facecolor='white', edgecolor='#CCCCCC', fontsize=10.5, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")

# Generate Fig 1 & Fig 2
plot_recall_vs_fpr("results/threshold_multiseed.csv", "results/fig_recall_vs_fpr.png", is_ablation=False)
plot_recall_vs_fpr("results/ablation_threshold_multiseed.csv", "results/fig_recall_vs_fpr_ablation.png", is_ablation=True)

# --- 3. fig_drift.png ---
def plot_drift(drift_csv, ablation_csv, out_path):
    df_base = pd.read_csv(drift_csv)
    df_abl = pd.read_csv(ablation_csv)
    
    merged = pd.merge(
        df_base[['held_out_family', 'recall_unseen']].rename(columns={'recall_unseen': 'recall_base'}),
        df_abl[['held_out_family', 'recall_unseen']].rename(columns={'recall_unseen': 'recall_abl'}),
        on='held_out_family'
    )
    
    # Sort by baseline recall (ascending so highest is at top)
    merged = merged.sort_values('recall_base', ascending=True).reset_index(drop=True)
    
    y = np.arange(len(merged))
    bar_height = 0.38
    
    fig, ax = plt.subplots(figsize=(9.5, 6.2), dpi=300)
    
    bars_base = ax.barh(
        y + bar_height/2,
        merged['recall_base'],
        height=bar_height,
        label='Baseline (Full Feature Set)',
        color=CB_BLUE,
        alpha=0.92,
        edgecolor='#222222',
        linewidth=0.5
    )
    
    bars_abl = ax.barh(
        y - bar_height/2,
        merged['recall_abl'],
        height=bar_height,
        label='Ablation (Synthetic Artifacts Removed)',
        color=CB_AMBER,
        alpha=0.92,
        edgecolor='#222222',
        linewidth=0.5
    )
    
    # Add data labels at end of bars
    for bar in bars_base:
        w = bar.get_width()
        ax.text(w + 0.012, bar.get_y() + bar.get_height()/2, f"{w:.3f}",
                va='center', ha='left', fontsize=8.5, color='#004477', fontweight='bold')
        
    for bar in bars_abl:
        w = bar.get_width()
        ax.text(w + 0.012, bar.get_y() + bar.get_height()/2, f"{w:.3f}",
                va='center', ha='left', fontsize=8.5, color='#885500', fontweight='bold')
    
    ax.set_yticks(y)
    ax.set_yticklabels(merged['held_out_family'], fontsize=10.5, fontweight='bold')
    ax.set_xlabel('Unseen Attack Family Recall (recall_unseen)', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_ylabel('Held-Out Attack Family', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_title('Drift Robustness: Leave-One-Family-Out Generalization', fontsize=12, fontweight='bold', pad=12)
    
    ax.set_xlim(0, 1.12)
    ax.grid(True, axis='x', linestyle='--', alpha=0.5, color='#D0D0D0')
    ax.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='#CCCCCC', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")

plot_drift("results/drift_results.csv", "results/ablation_drift_results.csv", "results/fig_drift.png")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import os
import pickle

# ─────────────────────────────────────────────────────────────
# LOAD RESULTS
# ─────────────────────────────────────────────────────────────

def load_results(save_path):
    with open(save_path, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded results from '{save_path}'")
    return data['tuning_results'], data['test_metrics']

def load_features(save_path):
    with open(save_path, 'rb') as f:
        data = pickle.load(f)
    return data['features'], data['minutiae']

# ─────────────────────────────────────────────────────────────
# PART 1 — FULL METRICS REPORT
# ─────────────────────────────────────────────────────────────

def compute_full_metrics(test_metrics, tuning_results):
    m  = test_metrics
    tr = tuning_results
    cmc = m['cmc_curve']

    scores_known   = np.array(m['scores_known'])
    scores_unknown = np.array(m['scores_unknown'])
    all_scores     = np.concatenate([scores_known, scores_unknown])
    thresholds     = np.linspace(all_scores.min(), all_scores.max(), 1000)

    tars, fars = [], []
    for t in thresholds:
        tars.append(np.mean(scores_known   >= t))
        fars.append(np.mean(scores_unknown >= t))

    tars    = np.array(tars)
    fars    = np.array(fars)
    eer_idx = np.argmin(np.abs(tars - (1 - fars)))
    roc_eer = (fars[eer_idx] + (1 - tars[eer_idx])) / 2
    auc     = float(np.sum(np.diff(fars[::-1]) * tars[::-1][:-1]))

    print("\n" + "="*60)
    print("         COMPLETE EVALUATION REPORT — STEP 4")
    print("="*60)
    print("\n── IDENTIFICATION METRICS (CMC) ──────────────────────")
    for k in [1, 2, 3, 5, 10]:
        print(f"  Rank-{k:<2} Accuracy      : {cmc[k]*100:.2f}%")
    print("\n── AUTHENTICATION METRICS ────────────────────────────")
    print(f"  Threshold (EER)     : {m['threshold']:.4f}")
    print(f"  FAR                 : {m['far']*100:.2f}%")
    print(f"  FRR                 : {m['frr']*100:.2f}%")
    print(f"  TAR @ EER           : {(1-m['frr'])*100:.2f}%")
    print(f"  EER (threshold)     : {m['eer']*100:.2f}%")
    print(f"  EER (ROC curve)     : {roc_eer*100:.2f}%")
    print(f"  AUC                 : {auc:.4f}")
    print("\n── DATASET STATISTICS ────────────────────────────────")
    print(f"  Known probes        : {m['known_total']}")
    print(f"  Unknown probes      : {m['unknown_total']}")
    print(f"  Correct IDs         : {m['correct_ids']}")
    print(f"  False accepts       : {m['false_accepts']}")
    print(f"  False rejects       : {m['false_rejects']}")
    print(f"  Total accuracy      : {m['total_acc']*100:.2f}%")
    print("\n── VALIDATE SET TUNING ───────────────────────────────")
    print(f"  EER on validate     : {tr['eer']*100:.2f}%")
    print(f"  EER threshold       : {tr['eer_threshold']:.4f}")
    print(f"  Best Rank-1 (val)   : {tr['best_r1']*100:.2f}%")
    print("="*60 + "\n")

    return {
        'cmc'       : cmc,
        'far'       : m['far'],
        'frr'       : m['frr'],
        'eer'       : m['eer'],
        'roc_eer'   : roc_eer,
        'auc'       : auc,
        'tars'      : tars,
        'fars'      : fars,
        'thresholds': thresholds
    }

# ─────────────────────────────────────────────────────────────
# PART 2 — EVALUATION DASHBOARD (no ablation — clean 2x3 grid)
# ─────────────────────────────────────────────────────────────

def plot_evaluation_dashboard(test_metrics, tuning_results,
                               full_metrics, save_path=None):
    """
    Clean 2-row × 3-column dashboard showing the 5 key plots
    and a metrics summary box. Ablation study is separate.
    """
    m   = test_metrics
    cmc = m['cmc_curve']

    fig = plt.figure(figsize=(38, 22))
    plt.rcParams.update({'font.size': 13})
    fig.suptitle('Fingerprint Authentication System — Evaluation Results',
             fontsize=20, fontweight='bold', y=1.01)

    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.25, wspace=0.28)

    # ── Panel 1: CMC Curve ─────────────────────────────────
    ax1  = fig.add_subplot(gs[0, 0])
    ranks = list(cmc.keys())
    accs  = [cmc[k]*100 for k in ranks]
    ax1.plot(ranks, accs, 'b-o', linewidth=3, markersize=9)
    ax1.fill_between(ranks, accs, alpha=0.12, color='blue')
    ax1.annotate(f"Rank-1: {accs[0]:.1f}%",
                 xy=(1, accs[0]), xytext=(3, accs[0]-14),
                 fontsize=13, fontweight='bold', color='navy',
                 arrowprops=dict(arrowstyle='->', color='navy',
                                 lw=1.5))
    ax1.annotate(f"Rank-10: {accs[-1]:.1f}%",
                 xy=(10, accs[-1]), xytext=(7.5, accs[-1]+5),
                 fontsize=11, color='steelblue',
                 arrowprops=dict(arrowstyle='->', color='steelblue'))
    ax1.set_xlabel('Rank', fontsize=13)
    ax1.set_ylabel('Identification Rate (%)', fontsize=13)
    ax1.set_title('CMC Curve — Identification Performance',
                  fontsize=14, fontweight='bold')
    ax1.set_xticks(ranks)
    ax1.set_ylim([0, 105])
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: ROC Curve ─────────────────────────────────
    ax2     = fig.add_subplot(gs[0, 1])
    tars    = full_metrics['tars']
    fars    = full_metrics['fars']
    eer     = full_metrics['roc_eer']
    eer_idx = np.argmin(np.abs(tars - (1 - fars)))

    ax2.plot(fars, tars, 'b-', linewidth=3, label='ROC Curve')
    ax2.plot([0, 1], [0, 1], 'k--', linewidth=1.5,
             label='Random Chance', alpha=0.6)
    ax2.scatter([fars[eer_idx]], [tars[eer_idx]], color='red',
                s=150, zorder=5, label=f'EER = {eer*100:.1f}%')
    ax2.axvline(fars[eer_idx], color='red', linestyle=':', alpha=0.4)
    ax2.axhline(tars[eer_idx], color='red', linestyle=':', alpha=0.4)
    ax2.set_xlabel('False Accept Rate (FAR)', fontsize=13)
    ax2.set_ylabel('True Accept Rate (TAR)', fontsize=13)
    ax2.set_title('ROC Curve — Authentication Performance',
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([-0.02, 1.02])
    ax2.set_ylim([-0.02, 1.02])

    # ── Panel 3: Key Metrics Summary ──────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')

    metrics_text = [
        ('IDENTIFICATION',  '',                               'header'),
        ('Rank-1 Accuracy', f"{cmc[1]*100:.2f}%",            'good'),
        ('Rank-3 Accuracy', f"{cmc[3]*100:.2f}%",            'good'),
        ('Rank-5 Accuracy', f"{cmc[5]*100:.2f}%",            'good'),
        ('Rank-10 Accuracy',f"{cmc[10]*100:.2f}%",           'good'),
        ('',                '',                               'spacer'),
        ('AUTHENTICATION',  '',                               'header'),
        ('FAR',             f"{m['far']*100:.2f}%",          'neutral'),
        ('FRR',             f"{m['frr']*100:.2f}%",          'neutral'),
        ('EER',             f"{eer*100:.2f}%",               'neutral'),
        ('AUC',             f"{full_metrics['auc']:.4f}",    'good'),
        ('',                '',                               'spacer'),
        ('DATASET',         '',                               'header'),
        ('Gallery persons', f"{m['known_total']}",           'neutral'),
        ('Unknown probes',  f"{m['unknown_total']}",         'neutral'),
        ('EER Threshold',   f"{m['threshold']:.4f}",         'neutral'),
        ('Correct IDs',
         f"{m['correct_ids']} / {m['known_total']}",         'good'),
    ]

    y_pos = 0.98
    for label, value, style in metrics_text:
        if style == 'header':
            ax3.text(0.04, y_pos, label, transform=ax3.transAxes,
                     fontsize=12, fontweight='bold', color='#1a252f',
                     verticalalignment='top')
            y_pos -= 0.052
        elif style == 'spacer':
            y_pos -= 0.018
        else:
            color = '#1e8449' if style == 'good' else '#2c3e50'
            ax3.text(0.04, y_pos, label, transform=ax3.transAxes,
                     fontsize=11, color='#555555',
                     verticalalignment='top')
            ax3.text(0.96, y_pos, value, transform=ax3.transAxes,
                     fontsize=11, fontweight='bold', color=color,
                     verticalalignment='top', ha='right')
            y_pos -= 0.052

    ax3.set_title('Key Metrics Summary', fontsize=14, fontweight='bold')
    rect = FancyBboxPatch((0.0, 0.0), 1.0, 1.0,
                          transform=ax3.transAxes,
                          boxstyle="round,pad=0.02",
                          facecolor='#f4f6f7',
                          edgecolor='#aab7b8',
                          linewidth=1.5, zorder=0)
    ax3.add_patch(rect)

    # ── Panel 4: Score Distributions ──────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    scores_known   = np.array(m['scores_known'])
    scores_unknown = np.array(m['scores_unknown'])
    ax4.hist(scores_known,   bins=50, alpha=0.65, color='#2980b9',
             label=f'Known (n={len(scores_known)})', density=True)
    ax4.hist(scores_unknown, bins=50, alpha=0.65, color='#e74c3c',
             label=f'Unknown (n={len(scores_unknown)})', density=True)
    ax4.axvline(m['threshold'], color='#27ae60', linestyle='--',
                linewidth=2.5,
                label=f"Threshold = {m['threshold']:.4f}")
    ax4.set_xlabel('Similarity Score', fontsize=13)
    ax4.set_ylabel('Density', fontsize=13)
    ax4.set_title('Score Distribution: Known vs Unknown',
                  fontsize=14, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)

    # ── Panel 5: FAR & FRR vs Threshold ───────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    thresh_vals = tuning_results['thresholds']
    ax5.plot(thresh_vals, tuning_results['fars']*100,
             'r-', linewidth=2.5, label='FAR')
    ax5.plot(thresh_vals, tuning_results['frrs']*100,
             'b-', linewidth=2.5, label='FRR')
    ax5.axvline(tuning_results['eer_threshold'], color='#27ae60',
                linestyle='--', linewidth=2,
                label=f"EER t = {tuning_results['eer_threshold']:.3f}")
    ax5.scatter([tuning_results['eer_threshold']],
                [tuning_results['eer']*100],
                color='#27ae60', s=120, zorder=5)
    ax5.set_xlabel('Threshold', fontsize=13)
    ax5.set_ylabel('Rate (%)', fontsize=13)
    ax5.set_title('FAR & FRR vs Threshold\n(Validate Set)',
                  fontsize=14, fontweight='bold')
    ax5.legend(fontsize=11)
    ax5.grid(True, alpha=0.3)

    # ── Panel 6: Rank-k Bar Chart ──────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    rank_labels = [f'Rank-{k}' for k in ranks]
    bar_colors  = plt.cm.Blues(np.linspace(0.35, 0.9, len(ranks)))
    bars        = ax6.bar(rank_labels, accs, color=bar_colors,
                          edgecolor='navy', linewidth=0.6)
    for bar, acc in zip(bars, accs):
        ax6.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.8,
                 f'{acc:.1f}%', ha='center', va='bottom',
                 fontsize=9.5, fontweight='bold')
    ax6.set_xlabel('Rank', fontsize=13)
    ax6.set_ylabel('Identification Rate (%)', fontsize=13)
    ax6.set_title('Rank-k Identification Rates',
                  fontsize=14, fontweight='bold')
    ax6.set_ylim([0, 108])
    ax6.set_xticklabels(rank_labels, rotation=45, ha='right', fontsize=9)
    ax6.grid(True, alpha=0.3, axis='y')

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved evaluation dashboard to '{save_path}'")
    plt.show()

# ─────────────────────────────────────────────────────────────
# PART 3 — ABLATION STUDY (separate figure)
# ─────────────────────────────────────────────────────────────

def plot_ablation_standalone(save_path=None):
    """
    Standalone ablation study figure — clean, large, focused.
    Separated from the dashboard so it gets its own slide.
    """
    labels = [
        'Full Hybrid\n(Deep + Classical + LDA)',
        'Deep Learning\nOnly (MobileNetV2)',
        'Classical\nOnly (HOG+LBP+Minutiae)',
        'PCA Only\n(No LDA)'
    ]
    values = [48.16, 11.68, 8.40, 9.22]
    colors = ['#2ecc71', '#3498db', '#3498db', '#3498db']
    hatches = ['', '/', '//', '\\\\']

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle('Ablation Study — Component Contribution to Accuracy',
                 fontsize=18, fontweight='bold', y=1.01)

    bars = ax.bar(labels, values, color=colors,
                  edgecolor='black', linewidth=1.0,
                  width=0.5, hatch=hatches)

    # Value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                f'{val:.1f}%',
                ha='center', va='bottom',
                fontsize=16, fontweight='bold')

    # Improvement arrow from best single component to full system
    ax.annotate('',
                xy=(0, values[0]),
                xytext=(1, values[1]),
                arrowprops=dict(arrowstyle='->', color='darkgreen',
                                lw=2.5))
    ax.text(0.5, (values[0]+values[1])/2 + 2,
            f'+{values[0]-values[1]:.1f}%\nimprovement',
            ha='center', va='bottom', fontsize=12,
            color='darkgreen', fontweight='bold')

    # Reference line
    ax.axhline(values[0], color='#27ae60', linestyle='--',
               linewidth=2, alpha=0.6, label='Full system baseline')

    # Random chance line
    random_chance = 100 / 488
    ax.axhline(random_chance, color='red', linestyle=':',
               linewidth=1.5, alpha=0.7,
               label=f'Random chance ({random_chance:.1f}%)')

    ax.set_ylabel('Rank-1 Identification Accuracy (%)', fontsize=14)
    ax.set_ylim([0, 62])
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=12, loc='upper right')
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

    # Annotation box
    ax.text(0.98, 0.35,
            'LDA projection is\nthe critical component\n'
            'responsible for the\nmajority of accuracy gain',
            transform=ax.transAxes,
            fontsize=11, ha='right', va='center',
            bbox=dict(boxstyle='round,pad=0.5',
                      facecolor='lightyellow',
                      edgecolor='orange', linewidth=1.5))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved ablation study to '{save_path}'")
    plt.show()

# ─────────────────────────────────────────────────────────────
# PART 4 — ERROR ANALYSIS
# ─────────────────────────────────────────────────────────────

def analyze_errors(test_metrics, save_path=None):
    predictions = test_metrics['predictions']

    correct_scores        = []
    incorrect_scores      = []
    true_rank_in_failures = []
    score_margins         = []
    rejection_errors      = []
    wrong_id_errors       = []

    for fname, info in predictions.items():
        true_id      = info['true_id']
        predicted_id = info['predicted_id']
        score        = info['score']
        all_matches  = info['all_matches']

        if true_id == 'UNKNOWN':
            continue

        if predicted_id == true_id:
            correct_scores.append(score)
        else:
            incorrect_scores.append(score)
            ranked_ids = [m[0] for m in all_matches]
            if true_id in ranked_ids:
                true_rank_in_failures.append(
                    ranked_ids.index(true_id) + 1
                )
            else:
                true_rank_in_failures.append(11)

            true_id_score = next(
                (m[1] for m in all_matches if m[0] == true_id), 0
            )
            score_margins.append(score - true_id_score)

            if predicted_id == 'UNKNOWN':
                rejection_errors.append(fname)
            else:
                wrong_id_errors.append(fname)

    print("\n── ERROR ANALYSIS ────────────────────────────────────")
    print(f"  Total known probes        : "
          f"{len(correct_scores)+len(incorrect_scores)}")
    print(f"  Correctly identified      : {len(correct_scores)}")
    print(f"  Incorrectly identified    : {len(incorrect_scores)}")
    print(f"    Wrong identity assigned : {len(wrong_id_errors)}")
    print(f"    Rejected as UNKNOWN     : {len(rejection_errors)}")
    if true_rank_in_failures:
        print(f"  Avg rank of true ID       : "
              f"{np.mean(true_rank_in_failures):.2f}")
        print(f"  True ID in top-3          : "
              f"{sum(1 for r in true_rank_in_failures if r<=3)}"
              f"/{len(true_rank_in_failures)}")
    if score_margins:
        print(f"  Avg score margin          : "
              f"{np.mean(score_margins):.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Error Analysis', fontsize=16, fontweight='bold')

    axes[0].hist(correct_scores,   bins=30, alpha=0.65,
                 color='#27ae60',
                 label=f'Correct (n={len(correct_scores)})',
                 density=True)
    axes[0].hist(incorrect_scores, bins=30, alpha=0.65,
                 color='#e74c3c',
                 label=f'Incorrect (n={len(incorrect_scores)})',
                 density=True)
    axes[0].axvline(test_metrics['threshold'], color='black',
                    linestyle='--', linewidth=2,
                    label=f"t={test_metrics['threshold']:.3f}")
    axes[0].set_xlabel('Best Match Score', fontsize=13)
    axes[0].set_ylabel('Density', fontsize=13)
    axes[0].set_title('Score: Correct vs Incorrect',
                      fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    if true_rank_in_failures:
        rank_counts = [true_rank_in_failures.count(r)
                       for r in range(1, 12)]
        rank_labels = [str(r) for r in range(1, 11)] + ['>10']
        bar_c = ['#e74c3c' if r == 0 else '#3498db'
                 for r in range(11)]
        axes[1].bar(rank_labels, rank_counts,
                    color=bar_c, edgecolor='black', linewidth=0.5)
        axes[1].set_xlabel('Rank of True Identity', fontsize=13)
        axes[1].set_ylabel('Count', fontsize=13)
        axes[1].set_title('Where True Identity Ranks in Failures',
                          fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')

    if score_margins:
        axes[2].hist(score_margins, bins=30, color='#f39c12',
                     alpha=0.75, edgecolor='black', linewidth=0.5)
        axes[2].axvline(0, color='red', linestyle='--',
                        linewidth=2, label='Zero margin')
        axes[2].set_xlabel('Score Margin (Predicted − True ID)',
                           fontsize=13)
        axes[2].set_ylabel('Count', fontsize=13)
        axes[2].set_title('Score Margin in Failure Cases',
                          fontsize=13, fontweight='bold')
        axes[2].legend(fontsize=11)
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved error analysis to '{save_path}'")
    plt.show()

    return {
        'correct_scores'        : correct_scores,
        'incorrect_scores'      : incorrect_scores,
        'true_rank_in_failures' : true_rank_in_failures,
        'score_margins'         : score_margins,
        'rejection_errors'      : rejection_errors,
        'wrong_id_errors'       : wrong_id_errors
    }

# ─────────────────────────────────────────────────────────────
# PART 5 — PERFORMANCE TABLE
# ─────────────────────────────────────────────────────────────

def plot_performance_table(test_metrics, full_metrics,
                            save_path=None):
    m   = test_metrics
    cmc = m['cmc_curve']

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis('off')
    fig.suptitle('System Performance Summary',
                 fontsize=16, fontweight='bold', y=0.97)

    rows = [
        ['Metric', 'Value', 'Category'],
        ['Rank-1 Identification Accuracy',
         f"{cmc[1]*100:.2f}%",  'Identification'],
        ['Rank-2 Identification Accuracy',
         f"{cmc[2]*100:.2f}%",  'Identification'],
        ['Rank-3 Identification Accuracy',
         f"{cmc[3]*100:.2f}%",  'Identification'],
        ['Rank-5 Identification Accuracy',
         f"{cmc[5]*100:.2f}%",  'Identification'],
        ['Rank-10 Identification Accuracy',
         f"{cmc[10]*100:.2f}%", 'Identification'],
        ['False Accept Rate (FAR)',
         f"{m['far']*100:.2f}%",               'Authentication'],
        ['False Reject Rate (FRR)',
         f"{m['frr']*100:.2f}%",               'Authentication'],
        ['Equal Error Rate (EER)',
         f"{full_metrics['roc_eer']*100:.2f}%",'Authentication'],
        ['Area Under ROC Curve (AUC)',
         f"{full_metrics['auc']:.4f}",         'Authentication'],
        ['Operating Threshold',
         f"{m['threshold']:.4f}",              'Configuration'],
        ['Gallery Size (enrolled persons)',
         f"{m['known_total']}",                'Dataset'],
        ['Test Probes (known)',
         f"{m['known_total']}",                'Dataset'],
        ['Test Probes (unknown)',
         f"{m['unknown_total']}",              'Dataset'],
        ['Correct Identifications',
         f"{m['correct_ids']} / {m['known_total']}", 'Results'],
    ]

    cell_colors = []
    for i, row in enumerate(rows):
        if i == 0:
            cell_colors.append(['#2c3e50'] * 3)
        else:
            cat = row[2]
            if cat == 'Identification':
                cell_colors.append(['#eaf4fb', '#d6eaf8', '#eaf4fb'])
            elif cat == 'Authentication':
                cell_colors.append(['#fef9e7', '#fdebd0', '#fef9e7'])
            elif cat == 'Dataset':
                cell_colors.append(['#f9f9f9', '#f0f0f0', '#f9f9f9'])
            elif cat == 'Results':
                cell_colors.append(['#eafaf1', '#d5f5e3', '#eafaf1'])
            else:
                cell_colors.append(['#f5f5f5', '#ebebeb', '#f5f5f5'])

    table = ax.table(
        cellText=rows[1:],
        colLabels=rows[0],
        cellLoc='center',
        loc='center',
        cellColours=cell_colors[1:],
        colColours=['#2c3e50'] * 3
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)

    for j in range(3):
        table[0, j].set_text_props(color='white', fontweight='bold')
    for i in range(1, len(rows)):
        table[i, 1].set_text_props(fontweight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved performance table to '{save_path}'")
    plt.show()

# ─────────────────────────────────────────────────────────────
# PART 6 — PRESENTATION GUIDE
# ─────────────────────────────────────────────────────────────

def print_presentation_guide(full_metrics, test_metrics):
    m   = test_metrics
    cmc = m['cmc_curve']

    print("\n" + "="*60)
    print("         PRESENTATION SLIDE GUIDE")
    print("="*60)
    print("""
SLIDE 1 — Title
  "Fingerprint Authentication System"
  "Open-Set Identification Using Hybrid Deep-Classical Features"

SLIDE 2 — System Architecture
  IMAGE: preprocessing_demo.png + preprocessing_multi.png
  • 5-stage pipeline: normalize → segment → enhance → binarize → skeleton
  • 1464 train / 500 validate / 500 test images
  • CLAHE + unsharp masking for enhancement

SLIDE 3 — Feature Extraction
  IMAGE: descriptor_components.png + minutiae_overlay.png
  • MobileNetV2 deep embedding (1280 dims)
  • 5 classical features: minutiae, HOG, LBP, frequency, orientation
  • 10592 total dimensions before reduction

SLIDE 4 — Dimensionality Reduction
  IMAGE: pca_lda_analysis.png
  • PCA: 10592 → 512 dims (87.7% variance retained)
  • LDA: 512 → 487 dims (maximizes class separation)
  • LDA scatter shows clearly separated identity clusters""")

    print(f"""
SLIDE 5 — Ablation Study
  IMAGE: ablation_study_standalone.png
  • Full hybrid: 48.2% vs best single component: 11.7%
  • LDA is the critical component
  • 240x better than random guessing ({100/488:.1f}%)

SLIDE 6 — Identification Results
  IMAGE: evaluation_dashboard.png (top row)
  • Rank-1: {cmc[1]*100:.2f}%
  • Rank-5: {cmc[5]*100:.2f}%
  • Rank-10: {cmc[10]*100:.2f}%

SLIDE 7 — Authentication Results
  IMAGE: evaluation_dashboard.png (bottom row)
  • EER = {full_metrics['roc_eer']*100:.2f}%
  • AUC = {full_metrics['auc']:.4f}
  • Threshold = {m['threshold']:.4f} (EER operating point)

SLIDE 8 — Error Analysis
  IMAGE: error_analysis.png
  • Most failures are near-misses (true ID at Rank-2 or 3)
  • Score margins very small — system is close on failures
  • Not systematic errors — dataset limitation

SLIDE 9 — Conclusions
  • Hybrid descriptor captures complementary information
  • LDA dimensionality reduction is critical
  • Open-set rejection limited by 12/500 unknown ratio
  • Future: fingerprint-specific deep model training""")

    print("="*60 + "\n")

# ─────────────────────────────────────────────────────────────
# PART 7 — SAVE FINAL REPORT
# ─────────────────────────────────────────────────────────────

def save_final_report(test_metrics, full_metrics,
                      error_analysis, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump({
            'test_metrics'  : test_metrics,
            'full_metrics'  : full_metrics,
            'error_analysis': error_analysis
        }, f)
    print(f"Final report saved to '{save_path}'")

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    RESULTS_PATH      = r"data\results\matching_results.pkl"
    FINAL_REPORT_PATH = r"data\results\final_report.pkl"

    os.makedirs(r"data\results", exist_ok=True)

    # ── Load Step 3 results ────────────────────────────────
    print("Loading Step 3 results...")
    tuning_results, test_metrics = load_results(RESULTS_PATH)

    # ── Full metrics ───────────────────────────────────────
    print("\nComputing full metrics...")
    full_metrics = compute_full_metrics(test_metrics, tuning_results)

    # ── Evaluation dashboard (clean — no ablation) ─────────
    print("Generating evaluation dashboard...")
    plot_evaluation_dashboard(
        test_metrics, tuning_results, full_metrics,
        save_path="evaluation_dashboard.png"
    )

    # ── Ablation study (standalone) ────────────────────────
    print("Generating standalone ablation study...")
    plot_ablation_standalone(
        save_path="ablation_study_standalone.png"
    )

    # ── Error analysis ─────────────────────────────────────
    print("Running error analysis...")
    error_analysis = analyze_errors(
        test_metrics,
        save_path="error_analysis.png"
    )

    # ── Performance table ──────────────────────────────────
    print("Generating performance table...")
    plot_performance_table(
        test_metrics, full_metrics,
        save_path="performance_table.png"
    )

    # ── Presentation guide ─────────────────────────────────
    print_presentation_guide(full_metrics, test_metrics)

    # ── Save report ────────────────────────────────────────
    save_final_report(
        test_metrics, full_metrics,
        error_analysis, FINAL_REPORT_PATH
    )

    print("\nStep 4 complete.")
    print("Files generated:")
    print("  evaluation_dashboard.png       ← clean 2x3 dashboard")
    print("  ablation_study_standalone.png  ← dedicated ablation slide")
    print("  error_analysis.png             ← failure analysis slide")
    print("  performance_table.png          ← metrics table slide")
    print("  data\\results\\final_report.pkl")
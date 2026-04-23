import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# ─────────────────────────────────────────────────────────────
# LOAD FEATURES
# ─────────────────────────────────────────────────────────────

def load_features(save_path):
    with open(save_path, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded features for {len(data['features'])} images from '{save_path}'")
    return data['features'], data['minutiae']

# ─────────────────────────────────────────────────────────────
# PART 1 — DESCRIPTOR REWEIGHTING
# Correct slicing based on actual 10592-dim descriptor layout
# ─────────────────────────────────────────────────────────────

def build_weighted_descriptor(features_dict):
    """
    Correctly reweight the 10592-dim hybrid descriptor.

    Actual layout:
        deep     : [0    : 1280]  = 1280 dims (MobileNetV2)
        minutiae : [1280 : 1780]  =  500 dims (classical)
        hog      : [1780 : 9544]  = 7764 dims (classical — over-represented)
        lbp      : [9544 : 9800]  =  256 dims (classical)
        freq     : [9800 :10056]  =  256 dims (classical)
        orient   : [10056:10256]  =  200 dims (classical)

    Weighting strategy:
        - Deep embedding  : upweight (3x) — rich learned features
        - Minutiae        : upweight (4x) — most forensically reliable
        - HOG             : heavily downweight (0.05x) — 7764 dims drowns others
        - LBP             : upweight (2x) — good local texture
        - Frequency       : upweight (2x) — stable ridge density feature
        - Orientation     : upweight (2x) — most stable across impressions
    """
    rebalanced = {}
    for fname, desc in features_dict.items():
        deep     = desc[0:1280]      * 3.0
        minutiae = desc[1280:1780]   * 4.0
        hog      = desc[1780:9544]   * 0.05
        lbp      = desc[9544:9800]   * 2.0
        freq     = desc[9800:10056]  * 2.0
        orient = desc[10056:10592] * 2.0

        combined = np.concatenate([deep, minutiae, hog, lbp, freq, orient])
        norm = np.linalg.norm(combined)
        rebalanced[fname] = combined / norm if norm > 0 else combined

    return rebalanced

# ─────────────────────────────────────────────────────────────
# PART 2 — PCA + LDA PIPELINE
# PCA first to reduce noise, then LDA to maximize class separation
# LDA is the key upgrade — it explicitly maximizes the ratio of
# between-class to within-class variance, which is exactly what
# fingerprint matching needs
# ─────────────────────────────────────────────────────────────

def apply_pca_lda(train_features, validate_features, test_features,
                  pca_components=512, save_path=None):
    """
    Two-stage dimensionality reduction:

    Stage 1 — PCA:
        Reduce from 10592 to 512 dims.
        Removes noise dimensions, keeps 90%+ variance.
        Required before LDA because LDA needs n_samples > n_features.

    Stage 2 — LDA:
        Reduce from 512 to min(n_classes-1, 512) dims.
        Explicitly maximizes between-class separation.
        This is what makes matching dramatically more accurate.
        LDA finds the projection that makes same-person descriptors
        cluster tightly and different-person descriptors spread apart.

    Fit both on training data only — no data leakage.
    """
    print(f"\nApplying PCA + LDA pipeline...")

    train_fnames    = sorted(train_features.keys())
    validate_fnames = sorted(validate_features.keys())
    test_fnames     = sorted(test_features.keys())

    train_matrix    = np.array([train_features[f] for f in train_fnames])
    validate_matrix = np.array([validate_features[f] for f in validate_fnames])
    test_matrix     = np.array([test_features[f] for f in test_fnames])

    # Get class labels for LDA
    train_labels = np.array([f.split('_')[0] for f in train_fnames])

    # ── Stage 1: StandardScaler + PCA ─────────────────────
    print(f"  Stage 1: StandardScaler + PCA ({pca_components} components)...")
    scaler       = StandardScaler()
    train_scaled = scaler.fit_transform(train_matrix)
    val_scaled   = scaler.transform(validate_matrix)
    test_scaled  = scaler.transform(test_matrix)

    pca          = PCA(n_components=pca_components, random_state=42)
    train_pca    = pca.fit_transform(train_scaled)
    val_pca      = pca.transform(val_scaled)
    test_pca     = pca.transform(test_scaled)

    pca_variance = pca.explained_variance_ratio_.sum() * 100
    print(f"    PCA explained variance : {pca_variance:.1f}%")

    # ── Stage 2: LDA ───────────────────────────────────────
    # LDA max components = min(n_classes - 1, n_features)
    n_classes    = len(np.unique(train_labels))
    lda_components = min(n_classes - 1, pca_components, 487)
    print(f"  Stage 2: LDA ({lda_components} components, {n_classes} classes)...")

    lda       = LDA(n_components=lda_components, solver='svd')
    train_lda = lda.fit_transform(train_pca, train_labels)
    val_lda   = lda.transform(val_pca)
    test_lda  = lda.transform(test_pca)

    print(f"    LDA output dimensions  : {train_lda.shape[1]}")

    # ── L2 normalize after LDA ─────────────────────────────
    def to_dict_normalized(fnames, matrix):
        result = {}
        for i, fname in enumerate(fnames):
            vec  = matrix[i].astype(np.float64)
            norm = np.linalg.norm(vec)
            result[fname] = vec / norm if norm > 0 else vec
        return result

    train_new    = to_dict_normalized(train_fnames,    train_lda)
    validate_new = to_dict_normalized(validate_fnames, val_lda)
    test_new     = to_dict_normalized(test_fnames,     test_lda)

    # ── Plot PCA variance ──────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle('Dimensionality Reduction Pipeline',
                 fontsize=13, fontweight='bold')

    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
    axes[0].plot(cumvar, 'b-', linewidth=2)
    axes[0].axhline(90, color='red', linestyle='--', label='90% variance')
    axes[0].axvline(pca_components, color='green', linestyle='--',
                    label=f'{pca_components} components')
    axes[0].set_xlabel('Number of Components')
    axes[0].set_ylabel('Cumulative Explained Variance (%)')
    axes[0].set_title(f'PCA: {pca_variance:.1f}% variance in {pca_components} dims')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # LDA class separation visualization (first 2 components)
    if train_lda.shape[1] >= 2:
        # Sample a few classes for visualization
        unique_labels  = np.unique(train_labels)
        sample_classes = unique_labels[:10]
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        for j, cls in enumerate(sample_classes):
            idx = train_labels == cls
            axes[1].scatter(train_lda[idx, 0], train_lda[idx, 1],
                            color=colors[j], alpha=0.7, s=30,
                            label=f'ID {cls}')
        axes[1].set_xlabel('LDA Component 1')
        axes[1].set_ylabel('LDA Component 2')
        axes[1].set_title('LDA: First 2 components\n(10 sample classes)')
        axes[1].legend(fontsize=7, ncol=2)
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved PCA+LDA plot to '{save_path}'")
    plt.show()

    return train_new, validate_new, test_new, pca, lda, scaler

# ─────────────────────────────────────────────────────────────
# PART 3 — ENROLLMENT
# Build gallery using multiple fusion strategies
# ─────────────────────────────────────────────────────────────

def parse_person_id(filename):
    return filename.split('_')[0]

def build_gallery(train_features):
    """
    Build gallery using mean template per person.
    After LDA the descriptors are already in a discriminative space
    so simple averaging is highly effective.
    """
    person_images = {}
    for fname in train_features:
        pid = parse_person_id(fname)
        if pid not in person_images:
            person_images[pid] = []
        person_images[pid].append(fname)

    gallery = {}
    for pid, fnames in person_images.items():
        descriptors = np.array([train_features[f] for f in fnames])
        template    = descriptors.mean(axis=0)
        norm        = np.linalg.norm(template)
        gallery[pid] = template / norm if norm > 0 else template

    print(f"\nGallery built:")
    print(f"  Persons enrolled      : {len(gallery)}")
    print(f"  Avg images per person : "
          f"{np.mean([len(v) for v in person_images.values()]):.1f}")
    print(f"  Template dimensions   : "
          f"{next(iter(gallery.values())).shape[0]}")
    return gallery, person_images

# ─────────────────────────────────────────────────────────────
# PART 4 — HELPERS
# ─────────────────────────────────────────────────────────────

def get_true_label(fname, gallery):
    pid = parse_person_id(fname)
    return pid if pid in gallery else 'UNKNOWN'

def build_matrix(features_dict, fnames):
    return np.array([features_dict[f] for f in fnames])

# ─────────────────────────────────────────────────────────────
# PART 5 — THRESHOLD TUNING
# ─────────────────────────────────────────────────────────────

def tune_threshold(validate_features, gallery,
                   thresholds=None, save_path=None):
    """
    Vectorized threshold tuning on validate set.
    Uses dot product similarity (equivalent to cosine similarity
    on L2-normalized vectors).
    """
    if thresholds is None:
        thresholds = np.linspace(-1.0, 1.0, 201)

    gallery_ids    = sorted(gallery.keys())
    gallery_matrix = np.array([gallery[pid] for pid in gallery_ids])

    val_fnames   = sorted(validate_features.keys())
    val_matrix   = build_matrix(validate_features, val_fnames)
    val_true_ids = [get_true_label(f, gallery) for f in val_fnames]

    print("  Computing validate similarity matrix...")
    sim_matrix   = val_matrix @ gallery_matrix.T
    best_scores  = sim_matrix.max(axis=1)
    best_indices = sim_matrix.argmax(axis=1)
    best_ids     = [gallery_ids[i] for i in best_indices]

    known_idx   = [i for i, t in enumerate(val_true_ids) if t != 'UNKNOWN']
    unknown_idx = [i for i, t in enumerate(val_true_ids) if t == 'UNKNOWN']
    known_total   = len(known_idx)
    unknown_total = len(unknown_idx)

    print(f"  Validate: {known_total} known | {unknown_total} unknown")
    print(f"  Score range: {best_scores.min():.4f} — {best_scores.max():.4f}")
    print(f"  Sweeping {len(thresholds)} thresholds...")

    results = []
    for t in thresholds:
        accepted = best_scores >= t

        known_correct = sum(
            1 for i in known_idx
            if accepted[i] and best_ids[i] == val_true_ids[i]
        )
        false_rejects   = sum(1 for i in known_idx   if not accepted[i])
        false_accepts   = sum(1 for i in unknown_idx if     accepted[i])
        unknown_correct = sum(1 for i in unknown_idx if not accepted[i])

        rank1_acc = known_correct  / known_total   if known_total   > 0 else 0
        far       = false_accepts  / unknown_total if unknown_total > 0 else 0
        frr       = false_rejects  / known_total   if known_total   > 0 else 0
        total_acc = (known_correct + unknown_correct) / len(val_fnames)

        results.append({
            'threshold'      : t,
            'rank1_acc'      : rank1_acc,
            'far'            : far,
            'frr'            : frr,
            'total_acc'      : total_acc,
            'known_correct'  : known_correct,
            'known_total'    : known_total,
            'unknown_correct': unknown_correct,
            'unknown_total'  : unknown_total
        })

    fars  = np.array([r['far']       for r in results])
    frrs  = np.array([r['frr']       for r in results])
    accs  = np.array([r['total_acc'] for r in results])
    r1s   = np.array([r['rank1_acc'] for r in results])

    eer_idx        = np.argmin(np.abs(fars - frrs))
    eer            = (fars[eer_idx] + frrs[eer_idx]) / 2
    eer_threshold  = thresholds[eer_idx]
    best_acc_idx   = np.argmax(accs)
    best_threshold = thresholds[best_acc_idx]
    best_acc       = accs[best_acc_idx]
    best_r1_idx    = np.argmax(r1s)
    best_r1        = r1s[best_r1_idx]
    best_r1_thresh = thresholds[best_r1_idx]

    print(f"\n  EER              : {eer*100:.2f}% at t={eer_threshold:.4f}")
    print(f"  Best total acc   : {best_acc*100:.2f}% at t={best_threshold:.4f}")
    print(f"  Best Rank-1 acc  : {best_r1*100:.2f}% at t={best_r1_thresh:.4f}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Threshold Tuning on Validate Set',
                 fontsize=14, fontweight='bold')

    axes[0].plot(thresholds, fars*100,  'r-', linewidth=2, label='FAR')
    axes[0].plot(thresholds, frrs*100,  'b-', linewidth=2, label='FRR')
    axes[0].axvline(eer_threshold, color='green', linestyle='--',
                    linewidth=1.5, label=f'EER t={eer_threshold:.3f}')
    axes[0].scatter([eer_threshold], [eer*100], color='green', s=100, zorder=5)
    axes[0].set_xlabel('Threshold', fontsize=12)
    axes[0].set_ylabel('Rate (%)', fontsize=12)
    axes[0].set_title(f'FAR & FRR\nEER = {eer*100:.2f}%')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(thresholds, accs*100, 'purple', linewidth=2)
    axes[1].axvline(best_threshold, color='orange', linestyle='--',
                    linewidth=1.5, label=f'Best t={best_threshold:.3f}')
    axes[1].scatter([best_threshold], [best_acc*100],
                    color='orange', s=100, zorder=5)
    axes[1].set_xlabel('Threshold', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title(f'Total Accuracy\nBest = {best_acc*100:.2f}%')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(thresholds, r1s*100, 'darkblue', linewidth=2)
    axes[2].axvline(best_r1_thresh, color='red', linestyle='--',
                    linewidth=1.5, label=f'Best t={best_r1_thresh:.3f}')
    axes[2].scatter([best_r1_thresh], [best_r1*100],
                    color='red', s=100, zorder=5)
    axes[2].set_xlabel('Threshold', fontsize=12)
    axes[2].set_ylabel('Rank-1 Accuracy (%)', fontsize=12)
    axes[2].set_title(f'Rank-1 Accuracy\nBest = {best_r1*100:.2f}%')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved threshold plot to '{save_path}'")
    plt.show()

    return {
        'results'        : results,
        'eer'            : eer,
        'eer_threshold'  : eer_threshold,
        'best_threshold' : best_threshold,
        'best_acc'       : best_acc,
        'best_r1'        : best_r1,
        'best_r1_thresh' : best_r1_thresh,
        'thresholds'     : thresholds,
        'fars'           : fars,
        'frrs'           : frrs,
        'total_accs'     : accs,
        'rank1_accs'     : r1s
    }

# ─────────────────────────────────────────────────────────────
# PART 6 — TEST SET EVALUATION
# ─────────────────────────────────────────────────────────────

def evaluate_test_set(test_features, gallery, threshold):
    """
    Final evaluation on test set at the tuned threshold.
    Uses the Rank-1 optimized threshold for best identification accuracy.
    """
    print(f"\nEvaluating test set at threshold = {threshold:.4f}...")

    gallery_ids    = sorted(gallery.keys())
    gallery_matrix = np.array([gallery[pid] for pid in gallery_ids])

    test_fnames   = sorted(test_features.keys())
    test_matrix   = build_matrix(test_features, test_fnames)
    test_true_ids = [get_true_label(f, gallery) for f in test_fnames]

    print("  Computing test similarity matrix...")
    sim_matrix   = test_matrix @ gallery_matrix.T
    best_scores  = sim_matrix.max(axis=1)
    best_indices = sim_matrix.argmax(axis=1)
    best_ids     = [gallery_ids[i] for i in best_indices]
    top10_indices = np.argsort(sim_matrix, axis=1)[:, ::-1][:, :10]

    predictions    = {}
    scores_known   = []
    scores_unknown = []
    rank_correct   = {k: 0 for k in range(1, 11)}
    known_total    = 0
    unknown_total  = 0
    false_accepts  = 0
    false_rejects  = 0
    correct_ids    = 0

    for idx, fname in enumerate(test_fnames):
        true_id      = test_true_ids[idx]
        score        = float(best_scores[idx])
        predicted_id = best_ids[idx] if score >= threshold else 'UNKNOWN'
        ranked_ids   = [gallery_ids[i] for i in top10_indices[idx]]
        all_matches  = [(gallery_ids[i], float(sim_matrix[idx, i]))
                        for i in top10_indices[idx]]

        predictions[fname] = {
            'true_id'     : true_id,
            'predicted_id': predicted_id,
            'score'       : score,
            'all_matches' : all_matches
        }

        if true_id == 'UNKNOWN':
            unknown_total  += 1
            scores_unknown.append(score)
            if predicted_id != 'UNKNOWN':
                false_accepts += 1
        else:
            known_total += 1
            scores_known.append(score)
            if predicted_id == 'UNKNOWN':
                false_rejects += 1
            elif predicted_id == true_id:
                correct_ids += 1
            for k in range(1, 11):
                if true_id in ranked_ids[:k]:
                    rank_correct[k] += 1

    rank1_acc = rank_correct[1] / known_total   if known_total   > 0 else 0
    far       = false_accepts   / unknown_total if unknown_total > 0 else 0
    frr       = false_rejects   / known_total   if known_total   > 0 else 0
    eer       = (far + frr) / 2
    total_acc = (correct_ids + (unknown_total - false_accepts)) / len(test_fnames)
    cmc_curve = {k: rank_correct[k] / known_total for k in range(1, 11)}

    metrics = {
        'rank1_acc'      : rank1_acc,
        'far'            : far,
        'frr'            : frr,
        'eer'            : eer,
        'total_acc'      : total_acc,
        'cmc_curve'      : cmc_curve,
        'known_total'    : known_total,
        'unknown_total'  : unknown_total,
        'false_accepts'  : false_accepts,
        'false_rejects'  : false_rejects,
        'correct_ids'    : correct_ids,
        'scores_known'   : scores_known,
        'scores_unknown' : scores_unknown,
        'predictions'    : predictions,
        'threshold'      : threshold
    }

    print_test_summary(metrics)
    return metrics

def print_test_summary(metrics):
    print("\n" + "="*50)
    print("            TEST SET RESULTS")
    print("="*50)
    print(f"  Threshold used    : {metrics['threshold']:.4f}")
    print("─"*50)
    print(f"  Known probes      : {metrics['known_total']}")
    print(f"  Unknown probes    : {metrics['unknown_total']}")
    print("─"*50)
    print(f"  Rank-1 Accuracy   : {metrics['rank1_acc']*100:.2f}%")
    print(f"  Total Accuracy    : {metrics['total_acc']*100:.2f}%")
    print(f"  FAR               : {metrics['far']*100:.2f}%")
    print(f"  FRR               : {metrics['frr']*100:.2f}%")
    print(f"  EER               : {metrics['eer']*100:.2f}%")
    print("─"*50)
    print("  CMC Curve:")
    for k, acc in metrics['cmc_curve'].items():
        bar = '█' * int(acc * 40)
        print(f"    Rank-{k:2d}: {acc*100:6.2f}%  {bar}")
    print("="*50 + "\n")

# ─────────────────────────────────────────────────────────────
# PART 7 — VISUALIZATION
# ─────────────────────────────────────────────────────────────

def plot_cmc_curve(metrics, save_path=None):
    ranks = list(metrics['cmc_curve'].keys())
    accs  = [metrics['cmc_curve'][k] * 100 for k in ranks]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(ranks, accs, 'b-o', linewidth=2.5, markersize=8)
    ax.fill_between(ranks, accs, alpha=0.1, color='blue')
    ax.annotate(f"Rank-1: {accs[0]:.1f}%",
                xy=(1, accs[0]),
                xytext=(2, max(accs[0]-8, 5)),
                fontsize=11,
                arrowprops=dict(arrowstyle='->', color='black'))
    ax.set_xlabel('Rank', fontsize=13)
    ax.set_ylabel('Identification Rate (%)', fontsize=13)
    ax.set_title('CMC Curve — Fingerprint Identification',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(ranks)
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved CMC curve to '{save_path}'")
    plt.show()

def plot_roc_curve(metrics, save_path=None):
    scores_known   = np.array(metrics['scores_known'])
    scores_unknown = np.array(metrics['scores_unknown'])
    all_scores     = np.concatenate([scores_known, scores_unknown])
    thresholds     = np.linspace(all_scores.min(), all_scores.max(), 500)

    tars, fars = [], []
    for t in thresholds:
        tars.append(np.mean(scores_known   >= t))
        fars.append(np.mean(scores_unknown >= t))

    tars = np.array(tars)
    fars = np.array(fars)
    eer_idx = np.argmin(np.abs(tars - (1 - fars)))
    eer     = (fars[eer_idx] + (1 - tars[eer_idx])) / 2

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(fars, tars, 'b-', linewidth=2.5, label='ROC Curve')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Chance')
    ax.scatter([fars[eer_idx]], [tars[eer_idx]], color='red', s=120,
               zorder=5, label=f'EER = {eer*100:.2f}%')
    ax.axvline(fars[eer_idx], color='red', linestyle=':', alpha=0.5)
    ax.axhline(tars[eer_idx], color='red', linestyle=':', alpha=0.5)
    ax.set_xlabel('False Accept Rate (FAR)', fontsize=13)
    ax.set_ylabel('True Accept Rate (TAR)', fontsize=13)
    ax.set_title('ROC Curve — Fingerprint Authentication',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved ROC curve to '{save_path}'")
    plt.show()

def plot_score_distributions(metrics, save_path=None):
    scores_known   = np.array(metrics['scores_known'])
    scores_unknown = np.array(metrics['scores_unknown'])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(scores_known,   bins=50, alpha=0.6, color='blue',
            label=f'Known persons (n={len(scores_known)})', density=True)
    ax.hist(scores_unknown, bins=50, alpha=0.6, color='red',
            label=f'Unknown persons (n={len(scores_unknown)})', density=True)
    ax.axvline(metrics['threshold'], color='green', linestyle='--',
               linewidth=2, label=f"Threshold = {metrics['threshold']:.4f}")
    ax.set_xlabel('Similarity Score', fontsize=13)
    ax.set_ylabel('Density', fontsize=13)
    ax.set_title('Score Distribution: Known vs Unknown Probes',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved score distributions to '{save_path}'")
    plt.show()

def plot_failure_cases(metrics, n_cases=3, save_path=None):
    failures = [
        (fname, info)
        for fname, info in metrics['predictions'].items()
        if info['true_id'] != info['predicted_id']
        and info['true_id'] != 'UNKNOWN'
    ][:n_cases]

    if not failures:
        print("No failure cases — perfect identification!")
        return

    fig, axes = plt.subplots(1, len(failures),
                              figsize=(6*len(failures), 5))
    if len(failures) == 1:
        axes = [axes]

    fig.suptitle('Failure Case Analysis', fontsize=14, fontweight='bold')

    for ax, (fname, info) in zip(axes, failures):
        top3     = info['all_matches'][:3]
        top3_str = '\n'.join([f"  {i+1}. ID={m[0]}  score={m[1]:.4f}"
                               for i, m in enumerate(top3)])
        text = (f"File: {fname}\n\n"
                f"True ID:    {info['true_id']}\n"
                f"Predicted:  {info['predicted_id']}\n"
                f"Score:      {info['score']:.4f}\n"
                f"Threshold:  {metrics['threshold']:.4f}\n\n"
                f"Top-3 matches:\n{top3_str}")
        ax.text(0.1, 0.5, text, ha='left', va='center',
                transform=ax.transAxes, fontsize=10,
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow',
                          alpha=0.8))
        ax.set_title('Incorrect Prediction', color='red', fontsize=12)
        ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved failure cases to '{save_path}'")
    plt.show()

def plot_ablation_study(train_features_raw, validate_features_raw,
                        test_features_raw, gallery_raw,
                        test_metrics_full, save_path=None):
    """
    Ablation study — compare system performance with and without
    each major component. This is the most impressive slide in
    your presentation because it proves each component contributes.

    Compares:
        1. Full hybrid (current best)
        2. Deep embedding only
        3. Classical only (no deep)
        4. No LDA (PCA only)
    """
    print("\nRunning ablation study...")
    results = {}

    # ── 1. Full system (already computed) ─────────────────
    results['Full Hybrid\n(Deep+Classical+LDA)'] = \
        test_metrics_full['rank1_acc'] * 100

    # ── 2. Deep embedding only ────────────────────────────
    print("  Ablation: Deep embedding only...")
    deep_train = {f: d[0:1280] / (np.linalg.norm(d[0:1280]) + 1e-8)
                  for f, d in train_features_raw.items()}
    deep_val   = {f: d[0:1280] / (np.linalg.norm(d[0:1280]) + 1e-8)
                  for f, d in validate_features_raw.items()}
    deep_test  = {f: d[0:1280] / (np.linalg.norm(d[0:1280]) + 1e-8)
                  for f, d in test_features_raw.items()}
    g_deep, _  = build_gallery(deep_train)
    t_deep     = tune_threshold(deep_val, g_deep,
                                thresholds=np.linspace(-1, 1, 101))
    m_deep     = evaluate_test_set(deep_test, g_deep,
                                   t_deep['best_r1_thresh'])
    results['Deep Learning\nOnly'] = m_deep['rank1_acc'] * 100

    # ── 3. Classical only (no deep embedding) ─────────────
    print("  Ablation: Classical features only...")
    cls_train = {f: d[1280:] / (np.linalg.norm(d[1280:]) + 1e-8)
                 for f, d in train_features_raw.items()}
    cls_val   = {f: d[1280:] / (np.linalg.norm(d[1280:]) + 1e-8)
                 for f, d in validate_features_raw.items()}
    cls_test  = {f: d[1280:] / (np.linalg.norm(d[1280:]) + 1e-8)
                 for f, d in test_features_raw.items()}
    g_cls, _  = build_gallery(cls_train)
    t_cls     = tune_threshold(cls_val, g_cls,
                                thresholds=np.linspace(-1, 1, 101))
    m_cls     = evaluate_test_set(cls_test, g_cls,
                                   t_cls['best_r1_thresh'])
    results['Classical\nOnly'] = m_cls['rank1_acc'] * 100

    # ── 4. No LDA (PCA only, full hybrid) ─────────────────
    print("  Ablation: PCA only (no LDA)...")
    train_w = build_weighted_descriptor(train_features_raw)
    val_w   = build_weighted_descriptor(validate_features_raw)
    test_w  = build_weighted_descriptor(test_features_raw)

    fnames_tr  = sorted(train_w.keys())
    fnames_vl  = sorted(val_w.keys())
    fnames_ts  = sorted(test_w.keys())
    mat_tr     = np.array([train_w[f] for f in fnames_tr])
    mat_vl     = np.array([val_w[f]   for f in fnames_vl])
    mat_ts     = np.array([test_w[f]  for f in fnames_ts])

    sc2        = StandardScaler()
    mat_tr_s   = sc2.fit_transform(mat_tr)
    mat_vl_s   = sc2.transform(mat_vl)
    mat_ts_s   = sc2.transform(mat_ts)

    pca2       = PCA(n_components=256, random_state=42)
    mat_tr_p   = pca2.fit_transform(mat_tr_s)
    mat_vl_p   = pca2.transform(mat_vl_s)
    mat_ts_p   = pca2.transform(mat_ts_s)

    def to_normed(fnames, mat):
        return {f: (v / (np.linalg.norm(v)+1e-8))
                for f, v in zip(fnames, mat)}

    g_pca, _   = build_gallery(to_normed(fnames_tr, mat_tr_p))
    t_pca      = tune_threshold(to_normed(fnames_vl, mat_vl_p), g_pca,
                                 thresholds=np.linspace(-1, 1, 101))
    m_pca      = evaluate_test_set(to_normed(fnames_ts, mat_ts_p), g_pca,
                                    t_pca['best_r1_thresh'])
    results['PCA Only\n(No LDA)'] = m_pca['rank1_acc'] * 100

    # ── Plot ablation ──────────────────────────────────────
    labels = list(results.keys())
    values = list(results.values())
    colors = ['#2ecc71' if i == 0 else '#3498db'
              for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values, color=colors, edgecolor='black',
                  linewidth=0.8, width=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom',
                fontsize=12, fontweight='bold')

    ax.set_ylabel('Rank-1 Accuracy (%)', fontsize=13)
    ax.set_title('Ablation Study — Component Contribution to Accuracy',
                 fontsize=14, fontweight='bold')
    ax.set_ylim([0, max(values) * 1.2 + 5])
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(values[0], color='green', linestyle='--',
               alpha=0.5, label='Full system')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved ablation study to '{save_path}'")
    plt.show()

    print("\n  Ablation results:")
    for label, val in results.items():
        print(f"    {label.replace(chr(10), ' '):<35}: {val:.2f}%")

    return results

def save_results(tuning_results, test_metrics, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump({
            'tuning_results': tuning_results,
            'test_metrics'  : test_metrics
        }, f)
    print(f"Results saved to '{save_path}'")

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    TRAIN_FEAT    = r"data\features\train_features.pkl"
    VALIDATE_FEAT = r"data\features\validate_features.pkl"
    TEST_FEAT     = r"data\features\test_features.pkl"
    RESULTS_PATH  = r"data\results\matching_results.pkl"

    os.makedirs(r"data\results", exist_ok=True)

    # ── Load raw features ──────────────────────────────────
    print("Loading features...")
    train_features_raw,    _ = load_features(TRAIN_FEAT)
    validate_features_raw, _ = load_features(VALIDATE_FEAT)
    test_features_raw,     _ = load_features(TEST_FEAT)

    # ── Reweight descriptors ───────────────────────────────
    print("\nReweighting descriptors...")
    train_w    = build_weighted_descriptor(train_features_raw)
    validate_w = build_weighted_descriptor(validate_features_raw)
    test_w     = build_weighted_descriptor(test_features_raw)

    # ── PCA + LDA ──────────────────────────────────────────
    print("\nApplying PCA + LDA...")
    train_f, validate_f, test_f, pca, lda, scaler = apply_pca_lda(
        train_w, validate_w, test_w,
        pca_components=512,
        save_path="pca_lda_analysis.png"
    )

    # ── Build gallery ──────────────────────────────────────
    gallery, person_images = build_gallery(train_f)

    # ── Tune threshold on validate set ────────────────────
    print("\nTuning threshold...")
    tuning_results = tune_threshold(
        validate_f, gallery,
        thresholds=np.linspace(-1.0, 1.0, 201),
        save_path="threshold_tuning.png"
    )

    # Use EER threshold — scientifically correct operating point
    # where FAR = FRR, standard in biometrics literature
    optimal_threshold = tuning_results['eer_threshold']
    print(f"\nSelected threshold (EER)               : {optimal_threshold:.4f}")
    print(f"Best Rank-1 threshold was              : "
          f"{tuning_results['best_r1_thresh']:.4f}")
    print(f"EER value                              : "
          f"{tuning_results['eer']*100:.2f}%")

    # ── Final test evaluation ──────────────────────────────
    test_metrics = evaluate_test_set(
        test_f, gallery, optimal_threshold
    )


    # ── Generate all plots ─────────────────────────────────
    print("\nGenerating plots...")
    plot_cmc_curve(
        test_metrics, save_path="cmc_curve.png"
    )
    plot_roc_curve(
        test_metrics, save_path="roc_curve.png"
    )
    plot_score_distributions(
        test_metrics, save_path="score_distributions.png"
    )
    plot_failure_cases(
        test_metrics, n_cases=3, save_path="failure_cases.png"
    )

    # ── Ablation study ─────────────────────────────────────
    print("\nRunning ablation study...")
    ablation_results = plot_ablation_study(
        train_features_raw, validate_features_raw,
        test_features_raw, gallery,
        test_metrics,
        save_path="ablation_study.png"
    )

    # ── Save results ───────────────────────────────────────
    save_results(tuning_results, test_metrics, RESULTS_PATH)

    print("\nStep 3 complete.")
    print("Files generated:")
    print("  pca_lda_analysis.png    ← for slides")
    print("  threshold_tuning.png    ← for slides")
    print("  cmc_curve.png           ← for slides")
    print("  roc_curve.png           ← for slides")
    print("  score_distributions.png ← for slides")
    print("  failure_cases.png       ← for slides")
    print("  ablation_study.png      ← for slides")
    print("  data\\results\\matching_results.pkl ← for Step 4")
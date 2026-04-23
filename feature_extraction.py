import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# ─────────────────────────────────────────────────────────────
# DEVICE SETUP
# ─────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ─────────────────────────────────────────────────────────────
# LOAD PREPROCESSED DATA
# ─────────────────────────────────────────────────────────────

def load_preprocessed(save_path):
    with open(save_path, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded {len(data)} preprocessed images from '{save_path}'")
    return data

# ─────────────────────────────────────────────────────────────
# PART 1 — DEEP LEARNING EMBEDDING (MobileNetV2)
# ─────────────────────────────────────────────────────────────

def load_mobilenet():
    """
    Load pretrained MobileNetV2 and strip the classification head.
    Outputs a 1280-dimensional embedding per image.
    No training needed — uses ImageNet pretrained weights.
    """
    model = models.mobilenet_v2(
        weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
    )
    model.classifier = torch.nn.Identity()
    model.eval()
    model.to(DEVICE)
    print("MobileNetV2 loaded successfully.")
    print(f"  Parameters : {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Mode       : inference only (no training)")
    print(f"  Output dim : 1280")
    return model

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def extract_deep_embedding(img_array, model):
    """
    Single-image deep embedding — used for visualization only.
    Full dataset uses the batched version in extract_features_from_dataset.
    """
    pil_img = Image.fromarray(img_array.astype(np.uint8))
    tensor  = TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        embedding = model(tensor)
    emb  = embedding.squeeze().cpu().numpy()
    norm = np.linalg.norm(emb)
    return emb / norm if norm > 0 else emb

# ─────────────────────────────────────────────────────────────
# PART 2 — MINUTIAE EXTRACTION (Classical)
# ─────────────────────────────────────────────────────────────

def compute_local_orientation(skeleton_bin, y, x, window=7):
    """
    Estimate local ridge orientation using gradient structure tensor.
    """
    half = window // 2
    rows, cols = skeleton_bin.shape
    y1, y2 = max(0, y-half), min(rows, y+half+1)
    x1, x2 = max(0, x-half), min(cols, x+half+1)
    patch = skeleton_bin[y1:y2, x1:x2].astype(np.float32)
    gx    = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
    gy    = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
    gxx   = np.sum(gx * gx)
    gyy   = np.sum(gy * gy)
    gxy   = np.sum(gx * gy)
    return 0.5 * np.arctan2(2 * gxy, gxx - gyy)

def extract_minutiae(skeleton, mask, border=20):
    """
    Vectorized crossing number minutiae detection.
    Detects ridge endings (CN=1) and bifurcations (CN=3).
    50x faster than pixel-by-pixel loop version.
    """
    rows, cols   = skeleton.shape
    skeleton_bin = (skeleton > 0).astype(np.uint8)

    p  = skeleton_bin
    p1 = p[0:-2, 0:-2]; p2 = p[0:-2, 1:-1]; p3 = p[0:-2, 2:]
    p4 = p[1:-1, 2:];   p5 = p[2:,   2:];   p6 = p[2:,   1:-1]
    p7 = p[2:,   0:-2]; p8 = p[1:-1, 0:-2]

    cn = (np.abs(p1.astype(int) - p2) + np.abs(p2.astype(int) - p3) +
          np.abs(p3.astype(int) - p4) + np.abs(p4.astype(int) - p5) +
          np.abs(p5.astype(int) - p6) + np.abs(p6.astype(int) - p7) +
          np.abs(p7.astype(int) - p8) + np.abs(p8.astype(int) - p1)) // 2

    center    = p[1:-1, 1:-1]
    mask_crop = mask[1:-1, 1:-1]

    ending_coords      = np.argwhere(
        (cn == 1) & (center == 1) & (mask_crop == 255)
    )
    bifurcation_coords = np.argwhere(
        (cn == 3) & (center == 1) & (mask_crop == 255)
    )

    minutiae = []
    for (y, x) in ending_coords:
        y, x = y+1, x+1
        if border < y < rows-border and border < x < cols-border:
            angle = compute_local_orientation(skeleton_bin, y, x)
            minutiae.append((x, y, angle, 0))

    for (y, x) in bifurcation_coords:
        y, x = y+1, x+1
        if border < y < rows-border and border < x < cols-border:
            angle = compute_local_orientation(skeleton_bin, y, x)
            minutiae.append((x, y, angle, 1))

    return minutiae

def filter_minutiae(minutiae, min_distance=10):
    """Remove spurious minutiae that are too close together."""
    if len(minutiae) == 0:
        return minutiae
    filtered = []
    used     = [False] * len(minutiae)
    for i, m1 in enumerate(minutiae):
        if used[i]:
            continue
        filtered.append(m1)
        for j, m2 in enumerate(minutiae):
            if i != j and not used[j]:
                dist = np.sqrt((m1[0]-m2[0])**2 + (m1[1]-m2[1])**2)
                if dist < min_distance:
                    used[j] = True
    return filtered

def minutiae_to_descriptor(minutiae, image_shape, num_points=100):
    """
    Convert minutiae list into a fixed-length normalized descriptor.
    Sorted by reliability: bifurcations first, then by distance from center.
    """
    rows, cols = image_shape
    descriptor = np.zeros((num_points, 5), dtype=np.float32)
    cx, cy     = cols / 2, rows / 2

    minutiae_sorted = sorted(
        minutiae,
        key=lambda m: (-(m[3]), np.sqrt((m[0]-cx)**2 + (m[1]-cy)**2))
    )
    for idx, (x, y, angle, mtype) in enumerate(minutiae_sorted[:num_points]):
        descriptor[idx, 0] = x / cols
        descriptor[idx, 1] = y / rows
        descriptor[idx, 2] = np.cos(angle)
        descriptor[idx, 3] = np.sin(angle)
        descriptor[idx, 4] = mtype

    return descriptor.flatten()  # shape: (500,)

# ─────────────────────────────────────────────────────────────
# PART 3 — CLASSICAL DESCRIPTORS
# ─────────────────────────────────────────────────────────────

def extract_hog_descriptor(img, target_size=(128, 128)):
    """
    HOG descriptor — captures ridge flow patterns at multiple scales.
    """
    img_resized = cv2.resize(img, target_size)
    hog = cv2.HOGDescriptor(
        _winSize=(128, 128),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9
    )
    return hog.compute(img_resized).flatten()  # shape: (1764,)

def extract_lbp_fast(img, target_size=(64, 64), num_bins=256):
    """
    Fully vectorized LBP — captures local ridge micro-texture.
    No loops at all — uses numpy roll for all 8 neighbors simultaneously.
    """
    img_small = cv2.resize(img, target_size).astype(np.float32)
    lbp       = np.zeros(target_size, dtype=np.uint8)

    for bit, (dy, dx) in enumerate([
        (-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)
    ]):
        shifted = np.roll(np.roll(img_small, dy, axis=0), dx, axis=1)
        lbp    += (shifted >= img_small).astype(np.uint8) * (2 ** bit)

    hist, _ = np.histogram(lbp.ravel(), bins=num_bins, range=(0, num_bins))
    hist    = hist.astype(np.float32)
    hist   /= (hist.sum() + 1e-7)
    return hist  # shape: (256,)

def extract_ridge_frequency_map(enhanced, mask, block_size=16):
    """
    Vectorized ridge frequency map using gradient magnitude per block.
    Each person has a unique ridge density pattern across their fingertip.
    """
    rows, cols  = enhanced.shape
    gx          = cv2.Sobel(enhanced.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy          = cv2.Sobel(enhanced.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    mag         = cv2.magnitude(gx, gy)
    n_rows      = rows // block_size
    n_cols      = cols // block_size
    mag_cropped = mag[:n_rows*block_size, :n_cols*block_size]
    freq_map    = mag_cropped.reshape(
        n_rows, block_size, n_cols, block_size
    ).mean(axis=(1, 3))
    freq_resized = cv2.resize(freq_map, (16, 16))
    freq_norm    = cv2.normalize(freq_resized, None, 0, 1, cv2.NORM_MINMAX)
    return freq_norm.flatten()  # shape: (256,)

def extract_orientation_map_descriptor(img, block_size=16):
    """
    Global ridge orientation map — most stable fingerprint feature
    across multiple impressions of the same finger.
    """
    rows, cols = img.shape
    gx = cv2.Sobel(img.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)

    orient_map = []
    for i in range(0, rows - block_size, block_size):
        for j in range(0, cols - block_size, block_size):
            gx_b  = gx[i:i+block_size, j:j+block_size]
            gy_b  = gy[i:i+block_size, j:j+block_size]
            gxx   = np.sum(gx_b**2)
            gyy   = np.sum(gy_b**2)
            gxy   = np.sum(gx_b * gy_b)
            theta = 0.5 * np.arctan2(2*gxy, gxx-gyy)
            orient_map.append(np.cos(2*theta))
            orient_map.append(np.sin(2*theta))

    orient_arr = np.array(orient_map, dtype=np.float32)
    target_len = 200
    if len(orient_arr) >= target_len:
        orient_arr = orient_arr[:target_len]
    else:
        orient_arr = np.pad(orient_arr, (0, target_len - len(orient_arr)))
    return orient_arr  # shape: (200,)

# ─────────────────────────────────────────────────────────────
# PART 4 — HYBRID DESCRIPTOR (single image — for visualization)
# ─────────────────────────────────────────────────────────────

def extract_hybrid_descriptor(preprocessed_result, model):
    """
    Single image hybrid descriptor — used for visualization only.
    Full dataset processing uses extract_features_from_dataset (batched).

    Components:
        MobileNetV2 embedding :  1280 dims  (deep learning)
        Minutiae descriptor   :   500 dims  (classical)
        HOG descriptor        :  1764 dims  (classical)
        LBP descriptor        :   256 dims  (classical)
        Ridge frequency map   :   256 dims  (classical)
        Orientation map       :   200 dims  (classical)
    ──────────────────────────────────────────────────────────
    Total                     :  4256 dims
    """
    skeleton    = preprocessed_result['skeleton']
    enhanced    = preprocessed_result['enhanced']
    mask        = preprocessed_result['mask']
    image_shape = skeleton.shape

    deep_emb      = extract_deep_embedding(enhanced, model)
    raw_minutiae  = extract_minutiae(skeleton, mask)
    raw_minutiae  = filter_minutiae(raw_minutiae, min_distance=10)
    minutiae_desc = minutiae_to_descriptor(raw_minutiae, image_shape, num_points=100)
    hog_desc      = extract_hog_descriptor(enhanced)
    lbp_desc      = extract_lbp_fast(enhanced)
    freq_desc     = extract_ridge_frequency_map(enhanced, mask)
    orient_desc   = extract_orientation_map_descriptor(enhanced)

    hybrid = np.concatenate([
        deep_emb, minutiae_desc, hog_desc,
        lbp_desc, freq_desc, orient_desc
    ])
    norm = np.linalg.norm(hybrid)
    if norm > 0:
        hybrid = hybrid / norm

    return hybrid, raw_minutiae

# ─────────────────────────────────────────────────────────────
# PART 5 — BATCHED FEATURE EXTRACTION (full dataset)
# ─────────────────────────────────────────────────────────────

def extract_features_from_dataset(preprocessed_data, model,
                                   save_path=None, batch_size=32):
    """
    Batched hybrid feature extraction for the full dataset.

    Strategy:
        1. Run MobileNetV2 in batches of 32 — 3-5x faster than one at a time
        2. Extract classical features per image (already fast/vectorized)
        3. Concatenate and L2-normalize the full hybrid descriptor
        4. Save to disk so Step 3 loads instantly

    Returns:
        features : dict { filename -> descriptor (4256,) }
        minutiae : dict { filename -> raw minutiae list }
    """
    features  = {}
    minutiae  = {}
    total     = len(preprocessed_data)
    filenames = sorted(preprocessed_data.keys())

    print(f"Extracting features from {total} images (batch_size={batch_size})...")

    # ── PHASE 1: Batched deep embeddings ──────────────────
    print("\n  Phase 1/2 — MobileNetV2 deep embeddings (batched)...")
    deep_embeddings = {}

    for batch_start in range(0, total, batch_size):
        batch_fnames  = filenames[batch_start:batch_start + batch_size]
        batch_tensors = []

        for fname in batch_fnames:
            img     = preprocessed_data[fname]['enhanced']
            pil_img = Image.fromarray(img.astype(np.uint8))
            tensor  = TRANSFORM(pil_img)
            batch_tensors.append(tensor)

        # Stack into one batch and run one forward pass
        batch = torch.stack(batch_tensors).to(DEVICE)
        with torch.no_grad():
            embeddings = model(batch).cpu().numpy()

        # L2 normalize each embedding
        for i, fname in enumerate(batch_fnames):
            emb  = embeddings[i]
            norm = np.linalg.norm(emb)
            deep_embeddings[fname] = emb / norm if norm > 0 else emb

        done = min(batch_start + batch_size, total)
        if done % 200 == 0 or done == total:
            print(f"    Deep embeddings: {done}/{total}")

    print(f"  Phase 1 complete — {len(deep_embeddings)} embeddings extracted.")

    # ── PHASE 2: Classical features per image ─────────────
    print("\n  Phase 2/2 — Classical feature extraction...")

    for i, fname in enumerate(filenames):
        try:
            result      = preprocessed_data[fname]
            skeleton    = result['skeleton']
            enhanced    = result['enhanced']
            mask        = result['mask']
            image_shape = skeleton.shape

            # Minutiae
            raw_mins      = extract_minutiae(skeleton, mask)
            raw_mins      = filter_minutiae(raw_mins, min_distance=10)
            minutiae_desc = minutiae_to_descriptor(
                raw_mins, image_shape, num_points=100
            )

            # Classical descriptors
            hog_desc    = extract_hog_descriptor(enhanced)
            lbp_desc    = extract_lbp_fast(enhanced)
            freq_desc   = extract_ridge_frequency_map(enhanced, mask)
            orient_desc = extract_orientation_map_descriptor(enhanced)

            # Combine with pre-computed deep embedding
            hybrid = np.concatenate([
                deep_embeddings[fname],
                minutiae_desc,
                hog_desc,
                lbp_desc,
                freq_desc,
                orient_desc
            ])

            # Final L2 normalization
            norm = np.linalg.norm(hybrid)
            if norm > 0:
                hybrid = hybrid / norm

            features[fname] = hybrid
            minutiae[fname] = raw_mins

        except Exception as e:
            print(f"    Warning: Failed on {fname}: {e}")

        if (i + 1) % 100 == 0 or (i + 1) == total:
            print(f"    Classical features: {i+1}/{total}")

    print(f"\n  Phase 2 complete — {len(features)}/{total} descriptors built.")
    print(f"  Descriptor dimensionality: {next(iter(features.values())).shape[0]}")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump({'features': features, 'minutiae': minutiae}, f)
        print(f"  Saved to '{save_path}'")

    return features, minutiae

def load_features(save_path):
    """Load previously saved features from disk."""
    with open(save_path, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded features for {len(data['features'])} images from '{save_path}'")
    return data['features'], data['minutiae']

# ─────────────────────────────────────────────────────────────
# PART 6 — VISUALIZATION FOR SLIDES
# ─────────────────────────────────────────────────────────────

def visualize_minutiae_overlay(preprocessed_result, model,
                                fname="sample", save_path=None):
    """
    Overlay detected minutiae on the original fingerprint.
    Red  = ridge endings
    Blue = bifurcations
    Lines show local orientation angle.
    Key visualization for your presentation slides.
    """
    _, raw_minutiae = extract_hybrid_descriptor(preprocessed_result, model)
    img       = preprocessed_result['original'].copy()
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    endings      = [(x,y,a) for x,y,a,t in raw_minutiae if t==0]
    bifurcations = [(x,y,a) for x,y,a,t in raw_minutiae if t==1]

    for (x, y, angle) in endings:
        cv2.circle(img_color, (x, y), 3, (0, 0, 255), -1)
        cv2.line(img_color, (x, y),
                 (x + int(8*np.cos(angle)), y + int(8*np.sin(angle))),
                 (0, 0, 200), 1)

    for (x, y, angle) in bifurcations:
        cv2.circle(img_color, (x, y), 3, (255, 100, 0), -1)
        cv2.line(img_color, (x, y),
                 (x + int(8*np.cos(angle)), y + int(8*np.sin(angle))),
                 (200, 80, 0), 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Minutiae Extraction — {fname}',
                 fontsize=14, fontweight='bold')

    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')

    axes[1].imshow(preprocessed_result['skeleton'], cmap='gray')
    axes[1].set_title('Skeleton')
    axes[1].axis('off')

    axes[2].imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
    axes[2].set_title(
        f'Minutiae Overlay\n'
        f'Endings: {len(endings)} (red)  |  '
        f'Bifurcations: {len(bifurcations)} (blue)'
    )
    axes[2].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved minutiae visualization to '{save_path}'")
    plt.show()

def visualize_descriptor_components(preprocessed_result, model,
                                     fname="sample", save_path=None):
    """
    Visualize all 5 descriptor components side by side.
    Includes deep embedding heatmap alongside classical components.
    Perfect for the Feature Extraction slide in your presentation.
    """
    enhanced = preprocessed_result['enhanced']

    # Deep embedding — shown as 32x40 heatmap
    deep_emb = extract_deep_embedding(enhanced, model)

    # HOG gradient magnitude
    img_resized = cv2.resize(enhanced, (128, 128))
    gx  = cv2.Sobel(img_resized, cv2.CV_32F, 1, 0)
    gy  = cv2.Sobel(img_resized, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)

    # LBP map
    img_small = cv2.resize(enhanced, (64, 64)).astype(np.float32)
    lbp_img   = np.zeros((64, 64), dtype=np.uint8)
    for bit, (dy, dx) in enumerate([
        (-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)
    ]):
        shifted  = np.roll(np.roll(img_small, dy, axis=0), dx, axis=1)
        lbp_img += (shifted >= img_small).astype(np.uint8) * (2 ** bit)

    # Orientation descriptor profile
    orient_desc = extract_orientation_map_descriptor(enhanced)

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.suptitle(f'Hybrid Descriptor Components — {fname}',
                 fontsize=14, fontweight='bold')

    axes[0].imshow(enhanced, cmap='gray')
    axes[0].set_title('Enhanced Input')
    axes[0].axis('off')

    axes[1].imshow(deep_emb.reshape(32, 40), cmap='viridis', aspect='auto')
    axes[1].set_title('MobileNetV2 Embedding\n(Deep Learning — 1280 dims)')
    axes[1].axis('off')

    axes[2].imshow(mag, cmap='hot')
    axes[2].set_title('HOG Gradient Magnitude\n(Ridge Flow — 1764 dims)')
    axes[2].axis('off')

    axes[3].imshow(lbp_img, cmap='nipy_spectral')
    axes[3].set_title('LBP Texture Map\n(Local Pattern — 256 dims)')
    axes[3].axis('off')

    axes[4].plot(orient_desc, color='steelblue', linewidth=0.8)
    axes[4].set_title('Orientation Profile\n(Ridge Direction — 200 dims)')
    axes[4].set_xlabel('Index')
    axes[4].set_ylabel('Value')
    axes[4].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved descriptor visualization to '{save_path}'")
    plt.show()

def print_feature_summary(train_features, validate_features, test_features):
    """Print a clean summary of extracted features."""
    sample = next(iter(train_features.values()))
    print("\n" + "="*52)
    print("          FEATURE EXTRACTION SUMMARY")
    print("="*52)
    print(f"  Descriptor dimensions  : {sample.shape[0]}")
    print(f"  Train features         : {len(train_features)}")
    print(f"  Validate features      : {len(validate_features)}")
    print(f"  Test features          : {len(test_features)}")
    print("─"*52)
    print("  Descriptor components:")
    print("    MobileNetV2 embedding : 1280 dims  ← deep learning")
    print("    Minutiae descriptor   :  500 dims  ← classical")
    print("    HOG descriptor        : 1764 dims  ← classical")
    print("    LBP descriptor        :  256 dims  ← classical")
    print("    Ridge frequency map   :  256 dims  ← classical")
    print("    Orientation map       :  200 dims  ← classical")
    print(f"    Total                 : {1280+500+1764+256+256+200} dims")
    print("="*52 + "\n")

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    TRAIN_PKL    = r"data\processed\train_data.pkl"
    VALIDATE_PKL = r"data\processed\validate_data.pkl"
    TEST_PKL     = r"data\processed\test_data.pkl"

    TRAIN_FEAT    = r"data\features\train_features.pkl"
    VALIDATE_FEAT = r"data\features\validate_features.pkl"
    TEST_FEAT     = r"data\features\test_features.pkl"

    os.makedirs(r"data\features", exist_ok=True)

    # ── Load MobileNetV2 once — reused for everything ──────
    print("Loading MobileNetV2...")
    model = load_mobilenet()

    # ── Load preprocessed data ─────────────────────────────
    print("\nLoading preprocessed data...")
    from preprocessing import load_preprocessed
    train_data    = load_preprocessed(TRAIN_PKL)
    validate_data = load_preprocessed(VALIDATE_PKL)
    test_data     = load_preprocessed(TEST_PKL)

    # ── Step 2A: Visualize on one image first ──────────────
    sample_key = sorted(train_data.keys())[0]
    print(f"\nVisualizing feature extraction on: {sample_key}")

    visualize_minutiae_overlay(
        train_data[sample_key], model,
        fname=sample_key,
        save_path="minutiae_overlay.png"
    )
    visualize_descriptor_components(
        train_data[sample_key], model,
        fname=sample_key,
        save_path="descriptor_components.png"
    )

    # ── Step 2B: Extract features from all datasets ────────
    print("\nExtracting features from all datasets...")
    train_features, train_minutiae = extract_features_from_dataset(
        train_data, model, save_path=TRAIN_FEAT, batch_size=32
    )
    validate_features, validate_minutiae = extract_features_from_dataset(
        validate_data, model, save_path=VALIDATE_FEAT, batch_size=32
    )
    test_features, test_minutiae = extract_features_from_dataset(
        test_data, model, save_path=TEST_FEAT, batch_size=32
    )

    # ── Step 2C: Summary ───────────────────────────────────
    print_feature_summary(train_features, validate_features, test_features)

    print("Step 2 complete. Features saved to data/features/")
    print("In Step 3, load with: train_features, _ = load_features(TRAIN_FEAT)")
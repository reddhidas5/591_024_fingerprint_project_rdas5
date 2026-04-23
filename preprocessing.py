import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from skimage.morphology import skeletonize
from scipy.ndimage import gaussian_filter

# ─────────────────────────────────────────────
# CORE PROCESSING FUNCTIONS
# ─────────────────────────────────────────────

def load_and_normalize(image_path):
    """Loads a BMP fingerprint image and apply CLAHE normalization."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)

def segment_fingerprint(img, block_size=16, threshold=0.1):
    """
    Removes background by variance thresholding.
    Blocks with low variance = background (empty space).
    Blocks with high variance = fingerprint ridges.
    """
    rows, cols = img.shape
    mask = np.zeros_like(img)
    for i in range(0, rows - block_size, block_size):
        for j in range(0, cols - block_size, block_size):
            block = img[i:i+block_size, j:j+block_size]
            if np.var(block) / 255.0 > threshold:
                mask[i:i+block_size, j:j+block_size] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    segmented = cv2.bitwise_and(img, img, mask=mask)
    return mask, segmented

def enhance_fingerprint(img, mask):
    """
    Enhance ridge clarity using unsharp masking:
    1. CLAHE for local contrast boost
    2. Unsharp masking to sharpen ridges
    3. Second CLAHE to even out brightness
    """
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_eq = clahe.apply(img)

    img_float = img_eq.astype(np.float32)
    blurred = gaussian_filter(img_float, sigma=2.0)
    sharpened = img_float + 1.5 * (img_float - blurred)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    clahe2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe2.apply(sharpened)
    enhanced = cv2.bitwise_and(enhanced, enhanced, mask=mask)
    return enhanced

def binarize(img, mask):
    """
    Convert enhanced image to black/white ridge map
    using adaptive thresholding.
    """
    img_blur = cv2.GaussianBlur(img, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        img_blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=15,
        C=6
    )
    binary = cv2.bitwise_and(binary, binary, mask=mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return binary

def skeletonize_image(binary_img):
    """Thin ridges to 1-pixel width for clean minutiae detection."""
    skeleton = skeletonize(binary_img > 0)
    return (skeleton * 255).astype(np.uint8)

# ─────────────────────────────────────────────
# FULL PIPELINE — SINGLE IMAGE
# ─────────────────────────────────────────────

def preprocess_fingerprint(image_path):
    """
    Run the full preprocessing pipeline on a single image.
    Returns a dict with all 5 stages + mask.
    """
    img       = load_and_normalize(image_path)
    mask, seg = segment_fingerprint(img)
    enhanced  = enhance_fingerprint(seg, mask)
    binary    = binarize(enhanced, mask)
    skeleton  = skeletonize_image(binary)
    return {
        'original':  img,
        'segmented': seg,
        'enhanced':  enhanced,
        'binary':    binary,
        'skeleton':  skeleton,
        'mask':      mask
    }

# ─────────────────────────────────────────────
# FULL PIPELINE — ENTIRE DATASET FOLDER
# ─────────────────────────────────────────────

def preprocess_dataset(data_folder, save_path=None):
    """
    Preprocess every .bmp image in a folder.
    Optionally saves results to disk as a .pkl file
    so you never have to reprocess again.

    Returns dict: { filename -> preprocessed result dict }
    """
    processed = {}
    image_files = sorted([f for f in os.listdir(data_folder) if f.endswith('.bmp')])
    total = len(image_files)
    print(f"Processing {total} images from '{data_folder}'...")

    for i, fname in enumerate(image_files):
        path = os.path.join(data_folder, fname)
        try:
            processed[fname] = preprocess_fingerprint(path)
        except Exception as e:
            print(f"  Warning: Failed on {fname}: {e}")
        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"  Progress: {i+1}/{total}")

    print(f"Done. Successfully processed {len(processed)}/{total} images.")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(processed, f)
        print(f"Saved preprocessed data to '{save_path}'")

    return processed

def load_preprocessed(save_path):
    """
    Loads previously saved preprocessed data from disk.

    """
    with open(save_path, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded {len(data)} preprocessed images from '{save_path}'")
    return data

# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

def visualize_single(image_path, save_path=None):
    """
    Show all 5 preprocessing stages for one image side by side.
    Use this to verify your pipeline looks correct.
    """
    results = preprocess_fingerprint(image_path)
    stages = [
        ('Original',  results['original']),
        ('Segmented', results['segmented']),
        ('Enhanced',  results['enhanced']),
        ('Binary',    results['binary']),
        ('Skeleton',  results['skeleton']),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle('Fingerprint Preprocessing Pipeline', fontsize=16, fontweight='bold')
    for ax, (title, img) in zip(axes, stages):
        ax.imshow(img, cmap='gray')
        ax.set_title(title, fontsize=12)
        ax.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to '{save_path}'")
    plt.show()
    return results

def visualize_multiple(data_folder, num_samples=3, save_path=None):
    """
    Show preprocessing results for multiple images at once.
    Useful for verifying consistency across different fingerprints.
    """
    image_files = sorted([f for f in os.listdir(data_folder) if f.endswith('.bmp')])
    samples = image_files[:num_samples]

    fig, axes = plt.subplots(num_samples, 5, figsize=(20, 4 * num_samples))
    fig.suptitle('Preprocessing Pipeline — Multiple Samples', fontsize=16, fontweight='bold')

    stage_titles = ['Original', 'Segmented', 'Enhanced', 'Binary', 'Skeleton']

    for row, fname in enumerate(samples):
        path = os.path.join(data_folder, fname)
        results = preprocess_fingerprint(path)
        stages = [results['original'], results['segmented'],
                  results['enhanced'], results['binary'], results['skeleton']]
        for col, (img, title) in enumerate(zip(stages, stage_titles)):
            ax = axes[row, col]
            ax.imshow(img, cmap='gray')
            if row == 0:
                ax.set_title(title, fontsize=12)
            ax.set_ylabel(fname[:12], fontsize=8)
            ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved multi-sample visualization to '{save_path}'")
    plt.show()

def print_dataset_summary(train_data, validate_data=None, test_data=None):
    """
    Print a summary of how many images were processed
    and how many unique persons are in each split.
    """
    def count_persons(data):
        persons = set(fname.split('_')[0] for fname in data.keys())
        return len(persons)

    print("\n" + "="*45)
    print("         DATASET SUMMARY")
    print("="*45)
    print(f"  Train     : {len(train_data):>5} images | {count_persons(train_data):>4} persons")
    if validate_data:
        print(f"  Validate  : {len(validate_data):>5} images | {count_persons(validate_data):>4} persons")
    if test_data:
        print(f"  Test      : {len(test_data):>5} images | {count_persons(test_data):>4} persons")
    print("="*45 + "\n")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":

    TRAIN_DIR    = r"data\train"
    VALIDATE_DIR = r"data\validate"
    TEST_DIR     = r"data\test"

    TRAIN_PKL    = r"data\processed\train_data.pkl"
    VALIDATE_PKL = r"data\processed\validate_data.pkl"
    TEST_PKL     = r"data\processed\test_data.pkl"

    # ── STEP 1A: Testing pipeline on one image first ──
    print("Testing pipeline on single image...")
    visualize_single(
        os.path.join(TRAIN_DIR, "000_R0_0.bmp"),
        save_path="preprocessing_demo.png"
    )

    # ── STEP 1B: Verifying consistency across 3 different images ──
    print("\nChecking consistency across multiple samples...")
    visualize_multiple(TRAIN_DIR, num_samples=3, save_path="preprocessing_multi.png")

    # ── STEP 1C: Processesing all three dataset splits ──
    print("\nProcessing full datasets...")
    train_data    = preprocess_dataset(TRAIN_DIR,    save_path=TRAIN_PKL)
    validate_data = preprocess_dataset(VALIDATE_DIR, save_path=VALIDATE_PKL)
    test_data     = preprocess_dataset(TEST_DIR,     save_path=TEST_PKL)

    # ── STEP 1D: Print summary ──
    print_dataset_summary(train_data, validate_data, test_data)

    print("Step 1 complete. Preprocessed data saved to data/processed/")
    
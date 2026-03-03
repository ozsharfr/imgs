import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import logging


# -------------------------
# Logging
# -------------------------
def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    return logging.getLogger(__name__)


logger = setup_logger()


# -------------------------
# Config
# -------------------------
DEFAULT_DATA_DIR = r"C:\Users\ozsha\Documents\el\data\set1"
DEFAULT_PDF = "registered_results.pdf"

ORB_NFEATURES = 1500
RANSAC_THRESH = 5.0
CLAHE_CLIP = 3.0
CLAHE_TILE = (10, 10)


# -------------------------
# PDF Export
# -------------------------
def save_images_to_pdf(image_list, scores=None, scores_before=None, pairs_names=None, output_filename=DEFAULT_PDF):
    """
    Saves up to 12 images in a 3x4 grid to an A4 PDF.
    Each cell title includes: Pair <id> and SSIM before -> after.
    """
    fig, axes = plt.subplots(4, 3, figsize=(8.27, 11.69))  # A4 size
    axes = axes.flatten()

    n = min(12, len(image_list))
    for i in range(12):
        ax = axes[i]
        if i < n:
            img = image_list[i]
            if img is None:
                ax.axis("off")
                continue

            # BGR -> RGB for Matplotlib
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            title_text = ""
            if pairs_names is not None and scores is not None and scores_before is not None:
                title_text = f"Pair {pairs_names[i]}\nSSIM: {scores_before[i]:.2f} -> {scores[i]:.2f}"

            ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
            ax.axis("off")
            if title_text:
                ax.set_title(title_text, fontsize=8)
        else:
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_filename, format="pdf")
    plt.close()
    logger.info(f"PDF saved: {output_filename}")


# -------------------------
# Registration pipeline
# -------------------------
def preprocess_for_registration(img: np.ndarray) -> np.ndarray:
    if img is None:
        return img
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
    return clahe.apply(img)


def register_images(img1: np.ndarray, img2: np.ndarray):
    """
    Registers img2 onto img1 using ORB + BFMatcher (crossCheck) + Homography (RANSAC).
    Returns registered image or None on failure.
    """
    if img1 is None or img2 is None:
        return None

    orb = cv2.ORB_create(ORB_NFEATURES)
    kp1, des1 = orb.detectAndCompute(preprocess_for_registration(img1), None)
    kp2, des2 = orb.detectAndCompute(preprocess_for_registration(img2), None)

    if des1 is None or des2 is None or kp1 is None or kp2 is None:
        return None
    if len(kp1) < 4 or len(kp2) < 4:
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    if not matches:
        return None

    matches = sorted(matches, key=lambda x: x.distance)

    src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)

    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_THRESH)
    if M is None:
        return None

    return cv2.warpPerspective(img2, M, (img1.shape[1], img1.shape[0]))


def visualize_registration(i1: np.ndarray, reg_i2: np.ndarray) -> np.ndarray:
    """
    Creates a red/green overlay image:
    - Red channel = i1
    - Green channel = reg_i2
    """
    if i1 is None or reg_i2 is None:
        return None

    overlay = np.zeros((i1.shape[0], i1.shape[1], 3), dtype=np.uint8)
    overlay[:, :, 2] = i1        # Red
    overlay[:, :, 1] = reg_i2    # Green
    return overlay


# -------------------------
# Evaluation
# -------------------------
def evaluate_dataset(data_dir: str):
    files = [f for f in os.listdir(data_dir) if f.endswith("_im_1.png")]
    if not files:
        logger.warning(f"No *_im_1.png files found in: {data_dir}")
        return [], [], [], []

    scores, scores_before, pairs_names, registered_images = [], [], [], []

    for f in files:
        pair_id = f.split("_")[0]
        p1 = os.path.join(data_dir, f)
        p2 = os.path.join(data_dir, f"{pair_id}_im_2.png")

        i1 = cv2.imread(p1, cv2.IMREAD_GRAYSCALE)
        i2 = cv2.imread(p2, cv2.IMREAD_GRAYSCALE)
        if i1 is None or i2 is None:
            logger.warning(f"Skipping {pair_id}: missing image(s)")
            continue

        reg_i2 = register_images(i1, i2)
        if reg_i2 is None:
            logger.warning(f"Skipping {pair_id}: registration failed")
            continue

        registered_images.append(visualize_registration(i1, reg_i2))

        # Keep behavior: plain SSIM on full image
        score_before = ssim(i1, i2)
        score_after = ssim(i1, reg_i2)

        scores_before.append(score_before)
        scores.append(score_after)
        pairs_names.append(pair_id)

        logger.info(f"Pair {pair_id} SSIM: {score_after:.4f} (before: {score_before:.4f})")

    if scores:
        logger.info(f"Average SSIM Score: {float(np.mean(scores)):.4f}")
        logger.info(f"Average SSIM Score Before Registration: {float(np.mean(scores_before)):.4f}")
    else:
        logger.warning("No successful registrations to score.")

    return registered_images, scores, scores_before, pairs_names


# -------------------------
# Entry point
# -------------------------
def main(data_dir: str = DEFAULT_DATA_DIR, pdf_out: str = DEFAULT_PDF):
    registered_images, scores, scores_before, pairs_names = evaluate_dataset(data_dir)
    # Take first 12 images for PDF - to better understand the results
    save_images_to_pdf(
        registered_images[:12],
        scores=scores[:12],
        scores_before=scores_before[:12],
        pairs_names=pairs_names[:12],
        output_filename=pdf_out,
    )


if __name__ == "__main__":
    main()
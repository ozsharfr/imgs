import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


# -------------------------
# Preprocessing
# -------------------------
def preprocess(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    img = cv2.GaussianBlur(img, (3, 3), 0)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(10, 10))
    return clahe.apply(img)


# -------------------------
# Registration
# -------------------------
def register_images(
    img_ref: np.ndarray,
    img_mov: np.ndarray,
    max_features: int = 4000,
    ratio: float = 0.75,
    ransac_thresh: float = 3.0,
    min_inliers: int = 15,
):
    """
    Registers img_mov to img_ref using ORB + RANSAC affine_partial.
    Returns: registered_img, success_flag
    """

    ref_p = preprocess(img_ref)
    mov_p = preprocess(img_mov)

    orb = cv2.ORB_create(nfeatures=max_features, fastThreshold=7)
    kp1, des1 = orb.detectAndCompute(ref_p, None)
    kp2, des2 = orb.detectAndCompute(mov_p, None)

    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return img_mov.copy(), False

    # KNN + Lowe ratio test
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good.append(m)

    if len(good) < 8:
        return img_mov.copy(), False

    pts_ref = np.float32([kp1[m.queryIdx].pt for m in good])
    pts_mov = np.float32([kp2[m.trainIdx].pt for m in good])

    M, inliers = cv2.estimateAffinePartial2D(
        pts_mov, pts_ref,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh
    )

    if M is None or inliers is None or inliers.sum() < min_inliers:
        return img_mov.copy(), False

    h, w = img_ref.shape[:2]
    registered = cv2.warpAffine(
        img_mov, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    return registered, True


# -------------------------
# Masked SSIM
# -------------------------
def masked_ssim(a, b):
    mask = (a > 0) & (b > 0)
    if mask.sum() < 100:
        return np.nan

    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1

    a_crop = a[y0:y1, x0:x1]
    b_crop = b[y0:y1, x0:x1]

    return ssim(a_crop, b_crop, data_range=255)


# -------------------------
# Dataset Evaluation
# -------------------------
def evaluate_dataset(data_dir: str):
    files = [f for f in os.listdir(data_dir) if f.endswith('_im_1.png')]

    scores_before = []
    scores_after = []

    for f in files:
        pair_id = f.split('_')[0]

        i1 = cv2.imread(os.path.join(data_dir, f), 0)
        i2 = cv2.imread(os.path.join(data_dir, f"{pair_id}_im_2.png"), 0)

        if i1 is None or i2 is None:
            continue

        reg_i2, ok = register_images(i1, i2)

        before = ssim(i1, i2, data_range=255)
        after = masked_ssim(i1, reg_i2)

        scores_before.append(before)
        scores_after.append(after)

        print(f"Pair {pair_id} | OK={ok} | "
              f"SSIM before={before:.4f} | after={after:.4f}")

    print("\n----------------------------------")
    print(f"Average SSIM Before: {np.nanmean(scores_before):.4f}")
    print(f"Average SSIM After : {np.nanmean(scores_after):.4f}")


# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    evaluate_dataset(r'C:\Users\ozsha\Documents\el\data\set1')
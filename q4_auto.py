import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import cv2
import math

def save_images_to_pdf(image_list, scores=None, scores_before=None, pairs_names=None, output_filename="registered_results.pdf"):
    # Create a figure with 4 rows and 3 columns (3x4 grid)
    fig, axes = plt.subplots(4, 3, figsize=(8.27, 11.69)) # A4 size
    axes = axes.flatten()

    for i in range(12):
        if i < len(image_list):
            img = image_list[i]
            # Convert BGR (OpenCV default) to RGB for Matplotlib
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            title_text = f"Pair {pairs_names[i]}\nSSIM: {scores_before[i]:.2f} -> {scores[i]:.2f}"
            axes[i].imshow(img, cmap='gray' if len(img.shape) == 2 else None)
            axes[i].axis('off') # Hide axes for clean look
            axes[i].set_title(title_text, fontsize=8)
        else:
            axes[i].axis('off') # Hide empty cells if fewer than 12

    plt.tight_layout()
    plt.savefig(output_filename, format='pdf')
    plt.close()
    print(f"PDF saved successfully: {output_filename}")

# Usage:
# Assuming 'registered_images' is a list of your 12 registered numpy arrays
# save_images_to_pdf(registered_images)


def preprocess_for_registration(img):
    # נרמול לטווח 0-255 אם התמונה לא בפורמט הזה
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
    # החלת CLAHE
    if True:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(10,10))
        img = clahe.apply(img)
    return img

def register_images(img1, img2):
    # 1. ORB Feature Extraction & Matching
    orb = cv2.ORB_create(1500)
    kp1, des1 = orb.detectAndCompute(preprocess_for_registration(img1), None)
    kp2, des2 = orb.detectAndCompute(preprocess_for_registration(img2), None)
    
    # 2. Match descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)
    
    # 3. Find Homography
    src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    # 4. Warp
    registered = cv2.warpPerspective(img2, M, (img1.shape[1], img1.shape[0]))
    return registered

def visualize_registration(i1, reg_i2):
    # 1. המרה ל-BGR כדי שנוכל להשתמש בערוצי צבע (כרגע התמונות הן Gray)
    i1_color = cv2.cvtColor(i1, cv2.COLOR_GRAY2BGR)
    reg_color = cv2.cvtColor(reg_i2, cv2.COLOR_GRAY2BGR)
    
    # 2. יצירת תמונה חדשה (שחורה בהתחלה)
    # נשתמש בערוץ האדום עבור התמונה המקורית (i1)
    # ובערוץ הירוק עבור התמונה הרשומה (reg_i2)
    overlay = np.zeros_like(i1_color)
    overlay[:, :, 2] = i1[:, :]  # ערוץ אדום (Red) מקבל את i1
    overlay[:, :, 1] = reg_i2[:, :] # ערוץ ירוק (Green) מקבל את reg_i2
    return overlay

def evaluate_dataset(data_dir):
    files = [f for f in os.listdir(data_dir) if f.endswith('_im_1.png')]
    scores = []
    scores_before = []
    pairs_names = [] 
    registered_images = []
    for f in files:
        pair_id = f.split('_')[0]
        i1 = cv2.imread(os.path.join(data_dir, f), 0)
        i2 = cv2.imread(os.path.join(data_dir, f"{pair_id}_im_2.png"), 0)
        
        reg_i2 = register_images(i1, i2)

        # Generate registered image for visualization
        
        registered_images.append(visualize_registration(i1, reg_i2))

        # Calculate SSIM before and after registration
        score_before = ssim(i1, i2)
        score = ssim(i1, reg_i2)
        scores.append(score)
        scores_before.append(score_before)
        pairs_names.append(pair_id)
        print(f"Pair {pair_id} SSIM: {score:.4f} (before: {score_before:.4f})")
        
    print(f"\nAverage SSIM Score: {np.mean(scores):.4f}")
    print(f"Average SSIM Score Before Registration: {np.mean(scores_before):.4f}")

    return registered_images , scores, scores_before , pairs_names

# הרצה
registered_images, scores, scores_before, pairs_names = evaluate_dataset(r'C:\Users\ozsha\Documents\el\data\set1')
save_images_to_pdf(registered_images[:12] , scores[:12], scores_before[:12], pairs_names[:12])
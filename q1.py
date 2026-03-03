import streamlit as st
import cv2
import numpy as np
import os

# הגדרת פונקציות העיבוד
def apply_clahe(img, clip_limit):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    return clahe.apply(img)

def apply_gaussian(img, sigma):
    return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)

def get_variance_map(img, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)
    mean = cv2.filter2D(img, -1, kernel)
    mean_sq = cv2.filter2D(img**2, -1, kernel)
    return mean_sq - mean**2

st.set_page_config(layout="wide")
st.title("Task 1: Inspection Pipeline (Red/Green Overlay)")

# Main folder
st.session_state.input_folder = st.sidebar.text_input("נתיב תיקיית המקור", value="C:\\Users\\ozsha\\Documents\\el\\data\\set1")
# 1. טעינת קבצים
uploaded_files = st.file_uploader("בחר תמונות מתיקיית set1", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])
print (f"Uploaded files: {[f.name for f in uploaded_files]}")  # Debugging line
if uploaded_files:
    # בחירת תמונה לניתוח
    file_names = [f.name for f in uploaded_files]
    st.session_state.selected_name = st.selectbox("בחר תמונה לניתוח", file_names)
    selected_file = next(f for f in uploaded_files if f.name == st.session_state.selected_name)
    
    # המרה לפורמט OpenCV
    file_bytes = np.asarray(bytearray(selected_file.read()), dtype=np.uint8)
    img_gray = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    
    if img_gray is not None:
        # --- תפריט צד (Sidebar) ---
        
        # שלב 1: Preprocessing
        st.sidebar.header("1. Preprocessing")
        st.session_state.prep_mode = st.sidebar.radio("שיטת עיבוד מקדים", ["None", "CLAHE", "Gaussian Blur"])
        
        processed_img = img_gray.copy()
        c_limit = 2.0  # Default
        g_sigma = 1.0  # Default
        
        if st.session_state.prep_mode == "CLAHE":
            st.session_state.c_limit = st.sidebar.slider("CLAHE Clip Limit", 1.0, 10.0, 1.83) # הערך מהצילום מסך שלך
            processed_img = apply_clahe(img_gray, st.session_state.c_limit)
        elif st.session_state.prep_mode == "Gaussian Blur":
            st.session_state.g_sigma = st.sidebar.slider("Gaussian Sigma", 0.1, 5.0, 1.0)
            processed_img = apply_gaussian(img_gray, st.session_state.g_sigma)

        # שלב 2: Segmentation (Variance)
        st.sidebar.header("2. Segmentation Settings")
        st.session_state.k_size = st.sidebar.slider("Kernel Size (Window)", 3, 51, 15, step=2) # הערך מהצילום מסך שלך
        st.session_state.var_thresh = st.sidebar.slider("Variance Threshold", 0, 5000, 423) # הערך מהצילום מסך שלך
        
        # חישוב המסיכה
        var_map = get_variance_map(processed_img.astype(np.float32), st.session_state.k_size)
        mask = (var_map < st.session_state.var_thresh).astype(np.uint8)
        # שלב 3: Morphology (אופציונלי)
        st.sidebar.header("3. Post-Processing")
        use_morph = st.sidebar.toggle("הפעל ניקוי מורפולוגי", value=True)
        m_size = 5 # Default
        if use_morph:
            st.session_state.m_size = st.sidebar.slider("Morphology Kernel Size", 3, 15, 5, step=2)
            m_kernel = np.ones((st.session_state.m_size, st.session_state.m_size), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, m_kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, m_kernel)

        # --- חדש: שלב 4: Batch Processing ---
        st.sidebar.markdown("---")
        st.sidebar.header("4. Batch Processing")
        out_folder = st.sidebar.text_input("שם תיקיית הפלט", value="set1_masks")

        if st.sidebar.button("Run on All Images"):
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)
            
            progress_bar = st.sidebar.progress(0)
            status_text = st.sidebar.empty()
            all_files = [f for f in os.listdir(st.session_state.input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
            print (f"Processing {len(all_files)} images...")
            for i, file_name in enumerate(all_files):
                file_path = os.path.join(st.session_state.input_folder, file_name)
                with open(file_path, "rb") as f:
                    f_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
                raw_img = cv2.imdecode(f_bytes, cv2.IMREAD_GRAYSCALE)

                # 2. החלת ה-Pipeline עם הפרמטרים מה-GUI
                if st.session_state.prep_mode == "CLAHE":
                    proc_img = apply_clahe(raw_img, st.session_state.c_limit)
                elif st.session_state.prep_mode == "Gaussian Blur":
                    proc_img = apply_gaussian(raw_img, st.session_state.g_sigma)
                else:
                    proc_img = raw_img

                v_map = get_variance_map(proc_img.astype(np.float32), st.session_state.k_size)
                batch_mask = (v_map < st.session_state.var_thresh).astype(np.uint8) * 255  # 0/255

                if use_morph:
                    kernel_m = np.ones((st.session_state.m_size, st.session_state.m_size), np.uint8)
                    batch_mask = cv2.morphologyEx(batch_mask, cv2.MORPH_CLOSE, kernel_m)
                    batch_mask = cv2.morphologyEx(batch_mask, cv2.MORPH_OPEN, kernel_m)

                # --- יצירת הויזואליזציה (Red/Green Overlay) ---
                alpha = st.session_state.get("alpha", 0.4)  # CHANGED: safe default if not set

                # CHANGED: background must be the ORIGINAL image, not the mask
                img_bgr = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2BGR)

                overlay = img_bgr.copy()

                # CHANGED: batch_mask is 0/255, so "smooth" is mask > 0 (NOT == 1)
                smooth = (batch_mask > 0)

                # CHANGED: OpenCV is BGR. Green=[0,180,0], Red=[0,0,180]
                overlay[smooth] = [0, 180, 0]     # smooth -> green
                overlay[~smooth] = [0, 0, 180]    # variable -> red

                final_view = cv2.addWeighted(
                    overlay, alpha,
                    img_bgr, 1 - alpha, 0
                )

                # 3. שמירה לדיסק
                save_path = os.path.join(out_folder, f"mask_{file_name}")
                cv2.imwrite(save_path, final_view)
                
                # עדכון התקדמות
                progress = (i + 1) / len(all_files)
                progress_bar.progress(progress)
                status_text.text(f"מעבד: {file_name}")
            
            st.sidebar.success(f"הסתיים! {len(all_files)} מסיכות נשמרו ב-{out_folder}")

        # --- יצירת הויזואליזציה (Red/Green Overlay) ---
        img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        overlay = img_bgr.copy()
        
        # 1 = חלק (ירוק), 0 = מורכב (אדום)
        overlay[mask == 1] = [0, 180, 0] 
        overlay[mask == 0] = [180, 0, 0] 
        
        st.session_state.alpha = st.sidebar.slider("Mask Opacity (Alpha)", 0.0, 1.0, 0.4)
        final_view = cv2.addWeighted(overlay, st.session_state.alpha, img_bgr, 1 - st.session_state.alpha, 0)

        # --- תצוגה ראשית ---
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(img_gray, use_container_width=True)
        with col2:
            st.subheader("Segmentation Overlay")
            st.image(final_view, use_container_width=True)
            
        st.caption("Green: Smooth regions (Value 1) | Red: Feature-rich regions (Value 0)")
        
    else:
        st.error("שגיאה בטעינת הקובץ.")
else:
    st.info("אנא העלה את התמונות מתיקיית set1 כדי להתחיל בכיול.")
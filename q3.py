
import os
import cv2
import numpy as np
import logging
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, GroupKFold, train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance

# -------------------------
# Config
# -------------------------
CONFIG = {
    "test_size": 0.25,
    "ksize": 3,
    "n_splits": 3,
    "random_state": 42,
    "param_grid": {
        "max_depth": [3, 5, 8],
        "min_samples_leaf": [5, 10],
        "n_estimators": [100]
    },
    "perm_repeats": 25,
    "report_pdf": "q3_report_one_page.pdf",
}

FEATURE_NAMES = [
    "mean_intensity",
    "std_intensity",
    "mean_sobel",
    "std_sobel",
    "laplacian_var",
    "fft_midfreq_mean",
]

CLASS_NAMES = ["cam_1", "cam_2"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


class CameraClassifier:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.scaler = StandardScaler()
        self.model = None

    def _get_features(self, img_path: str):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None

        # FFT feature
        fshift = np.fft.fftshift(np.fft.fft2(img))
        mag = 20 * np.log(np.abs(fshift) + 1.0)
        fft_feat = np.mean(
            mag[mag.shape[0] // 4: 3 * mag.shape[0] // 4,
                mag.shape[1] // 4: 3 * mag.shape[1] // 4]
        )

        # Sobel feature
        sobel = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=CONFIG["ksize"])

        return [
            float(np.mean(img)),
            float(np.std(img)),
            float(np.mean(sobel)),
            float(np.std(sobel)),
            float(cv2.Laplacian(img, cv2.CV_64F).var()),
            float(fft_feat),
        ]

    def prepare_data(self):
        files = [f for f in os.listdir(self.data_dir) if f.endswith(".png")]
        if not files:
            raise FileNotFoundError(f"No .png files found in: {self.data_dir}")

        unique_ids = list(set([f.split("_")[0] for f in files]))
        train_ids, test_ids = train_test_split(
            unique_ids,
            test_size=CONFIG["test_size"],
            random_state=CONFIG["random_state"],
        )

        def process(ids):
            X, y, groups = [], [], []
            for f in files:
                fid = f.split("_")[0]
                if fid not in ids:
                    continue
                feat = self._get_features(os.path.join(self.data_dir, f))
                if feat is None:
                    continue
                X.append(feat)
                y.append(0 if "1.png" in f else 1)  # original labeling convention
                groups.append(fid)
            return np.array(X), np.array(y), groups

        X_tr, y_tr, grp_tr = process(train_ids)
        X_te, y_te, grp_te = process(test_ids)
        return X_tr, y_tr, grp_tr, X_te, y_te, grp_te

    def train(self, X_train, y_train, groups):
        grid = GridSearchCV(
            RandomForestClassifier(random_state=CONFIG["random_state"]),
            CONFIG["param_grid"],
            cv=GroupKFold(n_splits=CONFIG["n_splits"]),
        )
        X_train_scaled = self.scaler.fit_transform(X_train)
        grid.fit(X_train_scaled, y_train, groups=groups)
        self.model = grid.best_estimator_
        logger.info(f"Best Params: {grid.best_params_}")

    def predict(self, X):
        Xs = self.scaler.transform(X)
        preds = self.model.predict(Xs)
        probs = self.model.predict_proba(Xs)[:, 1]
        return preds, probs

    def export_report_pdf_one_page(self, X_test, y_test, pdf_path: str = None):
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        if pdf_path is None:
            pdf_path = CONFIG["report_pdf"]

        preds, probs = self.predict(X_test)
        auc = roc_auc_score(y_test, probs)
        report_txt = classification_report(y_test, preds, target_names=CLASS_NAMES)
        cm = confusion_matrix(y_test, preds)

        # Regular importance (Gini)
        gini = np.asarray(self.model.feature_importances_, dtype=float)
        order_g = np.argsort(gini)[::-1]

        # Permutation importance (boxplots)
        Xs = self.scaler.transform(X_test)
        perm = permutation_importance(
            self.model,
            Xs,
            y_test,
            n_repeats=CONFIG["perm_repeats"],
            random_state=CONFIG["random_state"],
            scoring="roc_auc",
        )
        perm_imps = perm.importances
        order_p = np.argsort(perm_imps.mean(axis=1))[::-1]

        # --- One-page layout (A4 landscape) ---
        fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
        fig.suptitle("Q3 One-Page Report (Classical Camera Classifier)", fontsize=14)

        gs = fig.add_gridspec(
            nrows=2, ncols=3,
            width_ratios=[1.0, 1.3, 1.3],
            height_ratios=[1.0, 1.0],
            left=0.05, right=0.98, top=0.90, bottom=0.07,
            wspace=0.35, hspace=0.35
        )

        # (1) Confusion matrix (top-left)
        ax_cm = fig.add_subplot(gs[0, 0])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
        disp.plot(ax=ax_cm, colorbar=False, values_format="d")
        ax_cm.set_title("Confusion Matrix", fontsize=11)

        # (2) AUC + Best params (bottom-left)
        ax_auc = fig.add_subplot(gs[1, 0])
        ax_auc.axis("off")
        ax_auc.text(0.02, 0.90, f"AUC (ROC): {auc:.4f}", fontsize=14, weight="bold")
        ax_auc.text(0.02, 0.68, f"Permutation repeats: {CONFIG['perm_repeats']}", fontsize=10)
        ax_auc.text(0.02, 0.52, "Tip: AUC≈1 strong, AUC≈0.5 random", fontsize=10)

        # (3) Classification report (spans both rows, middle column)
        ax_rep = fig.add_subplot(gs[:, 1])
        ax_rep.axis("off")
        ax_rep.set_title("Classification Report", fontsize=11, pad=8)
        ax_rep.text(0.01, 0.98, report_txt, va="top", family="monospace", fontsize=8.5)

        # (4) Gini importance barplot (top-right)
        ax_gini = fig.add_subplot(gs[0, 2])
        ax_gini.bar([FEATURE_NAMES[i] for i in order_g], gini[order_g])
        ax_gini.set_title("Feature Importance (Gini)", fontsize=11)
        ax_gini.tick_params(axis="x", rotation=35, labelsize=8)
        ax_gini.set_ylabel("Importance", fontsize=9)

        # (5) Permutation importance boxplot (bottom-right)
        ax_perm = fig.add_subplot(gs[1, 2])
        data = [perm_imps[i] for i in order_p]
        labels = [FEATURE_NAMES[i] for i in order_p]
        ax_perm.boxplot(data, labels=labels, vert=True)
        ax_perm.set_title("Permutation Importance (ROC-AUC drop)", fontsize=11)
        ax_perm.tick_params(axis="x", rotation=35, labelsize=8)
        ax_perm.set_ylabel("Δ AUC", fontsize=9)

        # Save as single-page PDF
        with PdfPages(pdf_path) as pdf:
            pdf.savefig(fig)
        plt.close(fig)

        logger.info(f"Saved one-page PDF report: {pdf_path}")


if __name__ == "__main__":
    clf = CameraClassifier(r"C:\Users\ozsha\Documents\el\data\set1")
    X_tr, y_tr, grp_tr, X_te, y_te, _ = clf.prepare_data()
    clf.train(X_tr, y_tr, grp_tr)
    clf.export_report_pdf_one_page(X_te, y_te)

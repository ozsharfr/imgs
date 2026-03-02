import os
import cv2
import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, GroupKFold, train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# Configuration parameters for model tuning
CONFIG = {
    "test_size": 0.25,          # Proportion of unique IDs to reserve for testing
    "ksize": 3,                 # Sobel kernel size: 3 captures fine edges
    "n_splits": 3,              # GroupKFold folds: ensures IDs stay in one group
    "random_state": 42,         # Ensures reproducibility of random splits
    "param_grid": {
        "max_depth": [3, 5, 8],          # Prevents tree growth to reduce overfitting
        "min_samples_leaf": [5, 10],     # Forces tree nodes to have min samples; regularization
        "n_estimators": [100]            # Number of trees in the forest
    }
}

logging.basicConfig(level=logging.INFO)

class CameraClassifier:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.scaler = StandardScaler()
        self.model = None

    def _get_features(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None: return None
        
        # FFT analysis: captures sensor-specific frequency noise
        fshift = np.fft.fftshift(np.fft.fft2(img))
        mag = 20 * np.log(np.abs(fshift) + 1)
        fft_feat = np.mean(mag[mag.shape[0]//4:3*mag.shape[0]//4, mag.shape[1]//4:3*mag.shape[1]//4])
        
        # Sobel/Laplacian: captures sensor-specific sharpening/edge profiles
        sobel = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=CONFIG["ksize"])
        return [np.mean(img), np.std(img), np.mean(sobel), np.std(sobel), 
                cv2.Laplacian(img, cv2.CV_64F).var(), fft_feat]

    def prepare_data(self):
        files = [f for f in os.listdir(self.data_dir) if f.endswith('.png')]
        unique_ids = list(set([f.split('_')[0] for f in files]))
        train_ids, test_ids = train_test_split(unique_ids, test_size=CONFIG["test_size"], random_state=CONFIG["random_state"])
        
        def process(ids):
            X, y, groups = [], [], []
            for f in files:
                fid = f.split('_')[0]
                if fid in ids:
                    feat = self._get_features(os.path.join(self.data_dir, f))
                    if feat:
                        X.append(feat); y.append(0 if '1.png' in f else 1); groups.append(fid)
            return np.array(X), np.array(y), groups
            
        return process(train_ids) + process(test_ids)

    def train(self, X_train, y_train, groups):
        grid = GridSearchCV(
            RandomForestClassifier(random_state=CONFIG["random_state"]), 
            CONFIG["param_grid"], 
            cv=GroupKFold(n_splits=CONFIG["n_splits"])
        )
        grid.fit(self.scaler.fit_transform(X_train), y_train, groups=groups)
        self.model = grid.best_estimator_
        logging.info(f"Best Params: {grid.best_params_}")

    def evaluate(self, X_test, y_test):
        X_test_scaled = self.scaler.transform(X_test)
        print(classification_report(y_test, self.model.predict(X_test_scaled)))
        print(f"AUC: {roc_auc_score(y_test, self.model.predict_proba(X_test_scaled)[:, 1]):.4f}")

if __name__ == "__main__":
    clf = CameraClassifier(r'C:\Users\ozsha\Documents\el\data\set1')
    X_tr, y_tr, grp_tr, X_te, y_te, _ = clf.prepare_data()
    clf.train(X_tr, y_tr, grp_tr)
    clf.evaluate(X_te, y_te)
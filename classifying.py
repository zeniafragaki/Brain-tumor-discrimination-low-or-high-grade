# author : Zenia Fragaki
# date " 25/1/2025

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from skimage.io import imread
from skimage.transform import resize
from skimage.filters import gaussian
from skimage.color import rgb2gray
from scipy.stats import skew, kurtosis
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
import cv2
import time

# ----------- parameters ----------- #
SEED = 42
np.random.seed(SEED)
DIMENSION = 128  
DATA_PATH = r"your path "
DATA_PATH_LOW = os.path.join(DATA_PATH, "LOW GRADE")
DATA_PATH_HIGH = os.path.join(DATA_PATH, "HIGH GRADE")

if not os.path.exists(DATA_PATH_LOW):
    raise FileNotFoundError(f"The directory {DATA_PATH_LOW} does not exist.")
if not os.path.exists(DATA_PATH_HIGH):
    raise FileNotFoundError(f"The directory {DATA_PATH_HIGH} does not exist.")

low_images = [os.path.join(DATA_PATH_LOW, f) for f in os.listdir(DATA_PATH_LOW) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
high_images = [os.path.join(DATA_PATH_HIGH, f) for f in os.listdir(DATA_PATH_HIGH) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]

# ----------- load ----------- #
def load_images(data_path_low, data_path_high, dim):
    """Φόρτωση και κανονικοποίηση εικόνων."""
    low_images = [os.path.join(data_path_low, f) for f in os.listdir(data_path_low) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    high_images = [os.path.join(data_path_high, f) for f in os.listdir(data_path_high) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]

    images, labels = [], []
    for img_path in low_images:
        try:
            img = imread(img_path)
            img_resized = resize(img, (dim, dim, 3), anti_aliasing=True)
            img_filtered = gaussian(img_resized, sigma=1)  # Apply Gaussian filtering
            images.append(img_filtered)
            labels.append(0)
        except Exception as e:
            print(f"Error reading image {img_path}: {e}")

    for img_path in high_images:
        try:
            img = imread(img_path)
            img_resized = resize(img, (dim, dim, 3), anti_aliasing=True)
            img_filtered = gaussian(img_resized, sigma=1)  # Apply Gaussian filtering
            images.append(img_filtered)
            labels.append(1)
        except Exception as e:
            print(f"Error reading image {img_path}: {e}")

    print(f"Final dataset: {len(images)} images, Labels: {len(labels)}")
    return np.array(images), np.array(labels)

# ----------- features ----------- #
def extract_features(images):
    """features extraction."""
    features = []
    feature_names = []
    
    # initial
    plot_first_image = True
    
    for img in images:
        # Convert to grayscale
        gray_img = rgb2gray(img)
        
        # Otsu's method for segmentation
        _, otsu_img = cv2.threshold((gray_img * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        if plot_first_image:
            # grayscale image
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(gray_img, cmap='gray')
            plt.title('Original Grayscale Image')
            
            #  Otsu segmented image
            plt.subplot(1, 2, 2)
            plt.imshow(otsu_img, cmap='gray')
            plt.title('Otsu Segmented Image')
            plt.show()
            
            plot_first_image = False
        
        # First-order features
        first_order_features = [
            np.mean(otsu_img),
            np.std(otsu_img, ddof=1),
            skew(otsu_img.flatten(), bias=True),
            kurtosis(otsu_img.flatten(), fisher=False)
        ]
        first_order_names = ['mean', 'std', 'skewness', 'kurtosis']
        
        # Second-order co-occurrence matrix features
        levs = 15
        gray_img = levs * (gray_img / np.max(gray_img))
        image = np.asarray(gray_img, dtype=np.uint8)
        result = graycomatrix(image, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=levs + 1)
        coocurrence_features = [
            np.mean(graycoprops(result, prop='contrast')),
            np.max(graycoprops(result, prop='contrast')) - np.min(graycoprops(result, prop='contrast')),
            np.mean(graycoprops(result, prop='dissimilarity')),
            np.max(graycoprops(result, prop='dissimilarity')) - np.min(graycoprops(result, prop='dissimilarity')),
            np.mean(graycoprops(result, prop='energy')),
            np.max(graycoprops(result, prop='energy')) - np.min(graycoprops(result, prop='energy')),
            np.mean(graycoprops(result, prop='homogeneity')),
            np.max(graycoprops(result, prop='homogeneity')) - np.min(graycoprops(result, prop='homogeneity')),
            np.mean(graycoprops(result, prop='correlation')),
            np.max(graycoprops(result, prop='correlation')) - np.min(graycoprops(result, prop='correlation')),
            np.mean(graycoprops(result, prop='ASM')),
            np.max(graycoprops(result, prop='ASM')) - np.min(graycoprops(result, prop='ASM'))
        ]
        coocurrence_names = [
            'contrast_mean', 'contrast_range',
            'dissimilarity_mean', 'dissimilarity_range',
            'energy_mean', 'energy_range',
            'homogeneity_mean', 'homogeneity_range',
            'correlation_mean', 'correlation_range',
            'ASM_mean', 'ASM_range'
        ]
        
        # Local Binary Pattern features
        lbp = local_binary_pattern(gray_img, P=8, R=1, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 8 + 3), range=(0, 8 + 2))
        lbp_features = lbp_hist.astype("float")
        lbp_names = [f'LBP_{i+1}' for i in range(len(lbp_features))]
        
        # Combine all features
        feature_vector = np.concatenate([
            first_order_features, coocurrence_features, lbp_features
        ])
        feature_names = first_order_names + coocurrence_names + lbp_names
        
        features.append(feature_vector)
    
    # Print feature names and values in DataFrame format
    df_features = pd.DataFrame(features, columns=feature_names)
    print(df_features)
    
    return np.array(features)

#load 
X, y = load_images(DATA_PATH_LOW, DATA_PATH_HIGH, DIMENSION)

# Split 70-30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)

# train test
X_train_features = extract_features(X_train)
X_test_features = extract_features(X_test)

# normalization
scaler = StandardScaler()
X_train_features = scaler.fit_transform(X_train_features)
X_test_features = scaler.transform(X_test_features)

# ----------- train and test----------- #
classifiers = {
    "KNN": (KNeighborsClassifier(), {'n_neighbors': [3, 5, 7]}),
    "SVM": (SVC(probability=True), {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}),
    "Logistic Regression": (LogisticRegression(), {'C': [0.1, 1, 10]}),
    "Random Forest": (RandomForestClassifier(), {'n_estimators': [50, 100, 200]}),
    "XGBoost": (XGBClassifier(), {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]})
}

best_accuracy = 0
best_classifier_name = None
best_classifier = None
accuracies = []
roc_curves = {}
computation_times = []
cv_results = {}

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

for name, (clf, param_grid) in classifiers.items():
    start_time = time.time()
    grid_search = GridSearchCV(clf, param_grid, cv=kf, scoring='accuracy')
    grid_search.fit(X_train_features, y_train)
    end_time = time.time()
    computation_time = end_time - start_time
    computation_times.append((name, computation_time))
    
    best_clf = grid_search.best_estimator_
    y_pred = best_clf.predict(X_test_features)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append((name, accuracy))
    
    print(f"{name} Best Parameters: {grid_search.best_params_}")
    print(f"{name} Accuracy: {accuracy:.2f}")
    print(f"{name} Computation Time: {computation_time:.2f} seconds")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low Grade', 'High Grade'], yticklabels=['Low Grade', 'High Grade'])
    plt.title(f'{name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    
    # ROC curve
    if hasattr(best_clf, "predict_proba"):
        y_prob = best_clf.predict_proba(X_test_features)[:, 1]
    else:  # Use decision function for SVM
        y_prob = best_clf.decision_function(X_test_features)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    roc_curves[name] = (fpr, tpr, roc_auc)
    
    # Cross-validation scores
    cv_scores = cross_val_score(best_clf, X_train_features, y_train, cv=kf, scoring='accuracy')
    cv_results[name] = cv_scores
    
    # Update best classifier
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_classifier_name = name
        best_classifier = best_clf

# Print best classifier
print(f"Best Classifier: {best_classifier_name} with Accuracy: {best_accuracy:.2f}")

# Plot histogram of accuracies
classifier_names, classifier_accuracies = zip(*accuracies)
plt.figure(figsize=(10, 6))
plt.bar(classifier_names, classifier_accuracies, color='skyblue')
plt.xlabel('Classifier')
plt.ylabel('Accuracy')
plt.title('Classifier Accuracies')
plt.ylim(0, 1)
plt.show()

# Plot ROC curves
plt.figure(figsize=(10, 6))
for name, (fpr, tpr, roc_auc) in roc_curves.items():
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc='lower right')
plt.show()

# Plot computation times
classifier_names, classifier_times = zip(*computation_times)
plt.figure(figsize=(10, 6))
plt.bar(classifier_names, classifier_times, color='lightgreen')
plt.xlabel('Classifier')
plt.ylabel('Computation Time (seconds)')
plt.title('Classifier Computation Times')
plt.show()

# Plot cross-validation results
plt.figure(figsize=(10, 6))
for name, cv_scores in cv_results.items():
    plt.plot(range(1, kf.get_n_splits() + 1), cv_scores, marker='o', label=f'{name} (Mean = {np.mean(cv_scores):.2f})')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('Cross-Validation Scores')
plt.legend(loc='lower right')
plt.show()

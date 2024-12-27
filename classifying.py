# @Zenia Fragaki
#4/12/2024

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from skimage.io import imread
from skimage.transform import resize
from skimage.filters import gaussian
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ----------- Παράμετροι ----------- #
SEED = 42
np.random.seed(SEED)
DIMENSION = 128  # Ανάλυση εικόνας
DATA_PATH_LOW = r"C:\Users\zenia\OneDrive\Υπολογιστής\8ο_9o εξ\οπτικη μικροσκοπια εργασια ML\LOW GRADE"  # Διαδρομή για Low Grade
DATA_PATH_HIGH = r"C:\Users\zenia\OneDrive\Υπολογιστής\8ο_9o εξ\οπτικη μικροσκοπια εργασια ML\HIGH GRADE"  # Διαδρομή για High Grade

# ----------- Φόρτωση Εικόνων ----------- #
def load_images(data_path_low, data_path_high, dim):
    """Φόρτωση και κανονικοποίηση εικόνων."""
    low_images = [os.path.join(data_path_low, f) for f in os.listdir(data_path_low) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    high_images = [os.path.join(data_path_high, f) for f in os.listdir(data_path_high) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]

    # Debug prints to check the image paths
    print(f"Found {len(low_images)} low grade images.")
    print(f"Found {len(high_images)} high grade images.")

    images, labels = [], []
    for img_path in low_images:
        try:
            img = imread(img_path)
            img_resized = resize(img, (dim, dim, 3), anti_aliasing=True)
            img_filtered = gaussian(img_resized, sigma=1)  # Apply Gaussian filtering
            img_filtered = img_filtered / 255.0  # Κανονικοποίηση
            images.append(img_filtered)
            labels.append(0)
        except Exception as e:
            print(f"Error reading image {img_path}: {e}")

    for img_path in high_images:
        try:
            img = imread(img_path)
            img_resized = resize(img, (dim, dim, 3), anti_aliasing=True)
            img_filtered = gaussian(img_resized, sigma=1)  # Apply Gaussian filtering
            img_filtered = img_filtered / 255.0  # Κανονικοποίηση
            images.append(img_filtered)
            labels.append(1)
        except Exception as e:
            print(f"Error reading image {img_path}: {e}")

    print(f"Final dataset: {len(images)} images, Labels: {len(labels)}")
    return np.array(images), np.array(labels)

# Φόρτωση εικόνων
X, y = load_images(DATA_PATH_LOW, DATA_PATH_HIGH, DIMENSION)

# Debug prints to check the shapes of X and y
print(f"Shape of X: {X.shape}")
print(f"Shape of y: {y.shape}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Fit the data generator on the training data
datagen.fit(X_train)

# Flatten the images for the classifiers
X_train_flat = X_train.reshape((X_train.shape[0], -1))
X_test_flat = X_test.reshape((X_test.shape[0], -1))

# Standardize the data
scaler = StandardScaler()
X_train_flat = scaler.fit_transform(X_train_flat)
X_test_flat = scaler.transform(X_test_flat)

# Print features and labels
print("Training features and labels:")
print(X_train_flat)
print(y_train)

print("Testing features and labels:")
print(X_test_flat)
print(y_test)

# Initialize classifiers with hyperparameter grids
classifiers = {
    "KNN": (KNeighborsClassifier(), {'n_neighbors': [3, 5, 7]}),
    "SVM": (SVC(), {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}),
    "Logistic Regression": (LogisticRegression(), {'C': [0.1, 1, 10]}),
    "Random Forest": (RandomForestClassifier(), {'n_estimators': [50, 100, 200]}),
    "XGBoost": (XGBClassifier(), {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]})
}

# Train and evaluate each classifier with hyperparameter tuning
best_accuracy = 0
best_classifier_name = None
best_classifier = None

for name, (clf, param_grid) in classifiers.items():
    grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train_flat, y_train)
    best_clf = grid_search.best_estimator_
    y_pred = best_clf.predict(X_test_flat)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Best Parameters: {grid_search.best_params_}")
    print(f"{name} Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low Grade', 'High Grade'], yticklabels=['Low Grade', 'High Grade'])
    plt.title(f'{name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    
    # Update best classifier
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_classifier_name = name
        best_classifier = best_clf

print(f"Best Classifier: {best_classifier_name} with Accuracy: {best_accuracy:.2f}")

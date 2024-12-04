# @Zenia Fragaki
#4/12/2024

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import seaborn as sns
import xgboost as xgb
from skimage import io
from skimage.transform import resize
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data Augmentation
def loadData(X, y, path1, path2, D):
    dim = D
    N = len(y)
    XX = np.zeros((N, dim, dim, 3), float)

    #  ImageDataGenerator for Augmentation
    datagen = ImageDataGenerator(
        rotation_range=40,      # Flip
        width_shift_range=0.2,  # Shift
        height_shift_range=0.2, # Height shift
        shear_range=0.2,        #Shear
        zoom_range=0.2,         #zoom
        horizontal_flip=True,   # horizontial flip
        fill_mode='nearest'     
    )
    
    for i in range(N):
        path = path1 if y[i] == 0 else path2
        im = io.imread(path + '/' + X[i])
        
        if len(im.shape) == 2 or im.shape[2] == 1:  #For black and white images
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
        
        im = np.asarray(im, dtype="float")
        
        # Augmenting of images
        im = resize(im, (dim, dim, 3), anti_aliasing=False)
        im = np.uint8(im)
        
        # augmentation with ImageDataGenerator
        x = np.expand_dims(im, axis=0)
        augmented_images = datagen.flow(x, batch_size=1)

        # Adding to dataset
        augmented_image = next(augmented_images)[0].astype('uint8')
        XX[i, :, :, :] = augmented_image

    return XX

# Diff classifiers
def classify_images(X, y, q, algorithm_choice):
    X_selected = X[:, :q]
    X_selected = X_selected.reshape(X_selected.shape[0], -1)  # Flatten the images
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Choose algorithm
    if algorithm_choice == 0:
        model = KNeighborsClassifier(n_neighbors=5)
    elif algorithm_choice == 1:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif algorithm_choice == 2:
        model = LogisticRegression(random_state=42, max_iter=1000)
    elif algorithm_choice == 3:
        model = SVC(probability=True, random_state=42)
    elif algorithm_choice == 4:
        model = DecisionTreeClassifier(random_state=42)
    elif algorithm_choice == 5:
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    elif algorithm_choice == 6:
        model = AdaBoostClassifier(n_estimators=100, random_state=42)
    elif algorithm_choice == 7:
        model = GaussianNB()
    elif algorithm_choice == 8:
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    else:
        raise ValueError("Invalid algorithm choice.")

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else y_pred)
    roc_auc = auc(fpr, tpr)

    # Confusion Matrix and Classification Report
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    #Loss
    if hasattr(model, "predict_proba"):
        from sklearn.metrics import log_loss
        y_prob = model.predict_proba(X_test_scaled)
        loss = log_loss(y_test, y_prob)
    else:
        loss = None  #No loss

    return accuracy, loss, roc_auc, fpr, tpr, cm, report, model

#Results
def summarize_results(results_summary):
    best_result = max(results_summary, key=lambda x: x[2])
    best_images, best_model, best_accuracy = best_result
    model_names = {
        0: "KNeighborsClassifier",
        1: "RandomForestClassifier",
        2: "LogisticRegression",
        3: "SVC",
        4: "DecisionTreeClassifier",
        5: "GradientBoostingClassifier",
        6: "AdaBoostClassifier",
        7: "GaussianNB",
        8: "XGBClassifier"
    }
    best_model_name = model_names[best_model]
    
    results_summary_df = pd.DataFrame(results_summary, columns=['Number of Images', 'Best Model', 'Best Accuracy'])
    
    summary = (
        f"Best Algorithm: {best_model_name}\n"
        f"Number of Images Used: {best_images}\n"
        f"Best Accuracy: {best_accuracy * 100:.2f}%\n\n"
        f"{results_summary_df.to_string(index=False)}"
    )
    
    return summary, results_summary_df

#Initialize
seed = 7
np.random.seed(seed)

#Dimensiona
Dimension = 96
maxImages_list = [30, 50, 60, 100]

grade = ["lowGrade", "highGrade"]
dataChoice = 0
results_summary = []

#Paths
path1 = r"C:\Users\zenia\OneDrive\Υπολογιστής\8ο_9o εξ\οπτικη μικροσκοπια εργασια ML\LOW GRADE"
path2 = r"C:\Users\zenia\OneDrive\Υπολογιστής\8ο_9o εξ\οπτικη μικροσκοπια εργασια ML\HIGH GRADE"

#Training
for maxImages in maxImages_list:
    X1 = os.listdir(path1)[:maxImages]
    X2 = os.listdir(path2)[:maxImages]
    
    y1 = np.zeros(len(X1), int)
    y2 = np.ones(len(X2), int)
    
    Xx = np.concatenate((X1, X2), axis=0)
    y = np.concatenate((y1, y2), axis=0)
    
    #Loading Data
    X = loadData(Xx, y, path1, path2, Dimension)
    
    best_accuracy = 0
    best_model = None
    all_results = []
    confusion_matrices = []
    roc_data = []
    losses = []

    # Classifying
    for B_model in range(9):  #
        accuracy, loss, auc_value, fpr, tpr, cm, report, model = classify_images(X, y, q=50, algorithm_choice=B_model)
        all_results.append([model.__class__.__name__, accuracy, auc_value, loss])  # Αποθήκευση αποτελεσμάτων
        confusion_matrices.append(cm)
        roc_data.append((fpr, tpr, auc_value))
        losses.append(loss)

        #Priniting
        print(f"Model: {model.__class__.__name__}")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"Loss: {loss if loss is not None else 'N/A'}")
        print(f"AUC: {auc_value:.2f}")
        print(f"Confusion Matrix:\n{cm}")
        print(f"Classification Report:\n{report}\n")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = B_model

    results_summary.append([maxImages, best_model, best_accuracy])

#Result summury
summary, results_summary_df = summarize_results(results_summary)
print("\nBest Model Summary:")
print(summary)

# 1Accuracy
model_names = ["KNeighbors", "Random Forest", "Logistic Regression", "SVC", "Decision Tree", "Gradient Boosting", "AdaBoost", "GaussianNB", "XGBoost"]
accuracies = [result[1] for result in all_results]
losses = [result[3] for result in all_results]

plt.figure(figsize=(10, 6))
plt.bar(model_names, accuracies)
plt.title('Accuracy of Different Models')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.show()

# 2. ROC Curve
plt.figure(figsize=(10, 6))
for i, (fpr, tpr, auc_value) in enumerate(roc_data):
    plt.plot(fpr, tpr, label=f'{model_names[i]} (AUC = {auc_value:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve for Different Models')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()

# 3. Confusion Matrix Heatmap
for i, cm in enumerate(confusion_matrices):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix for {model_names[i]}")
    plt.show()


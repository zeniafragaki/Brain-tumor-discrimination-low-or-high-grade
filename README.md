# Brain-tumor-discrimination-low-or-high-grade

 Part of a project of Optical Microscopy and Biological Images Analysis

 Statistical analysis of features able to discriminate between low and high grade brain tumor H&amp;E images

 This work is discussing the use of Machine Learning (ML) techniques for the classification of low- and high-grade brain tumor images. 
 Some of the most popular classifying algorithms were used to test the research case. Our case study has selected K-Nearest Neighbors (KNN), Logistic Regression, Support Vector Machine (SVM), Decision Trees, Random 
 Forest, Gradient Boosting, and XGBoost classifiers. Because of our limited dataset which contains 20 images (10 low and 10 high grade brain tumor), data augmentation techniques were used to achieve higher 
 statistic accuracy values. 
 Among our classifier’s algorithms, Logistic Regression scored the best accuracy 83.33% with a data set of 50 images. Below will be presented more specifically the highlights of this study and possible future 
 corrections and adjustments for better results


# Brain tumor


A brain tumor is an abnormal mass of cells that forms in or around the brain, disrupting its normal functions. Tumors can be benign (non-cancerous) or malignant (cancerous) and are classified based on their origin, growth rate, and behavior.
Primary brain tumors originate in the brain, while secondary brain tumors, or metastases, spread from cancers in other parts of the body. Tumors are further categorized by grade, ranging from low-grade (slow growing, less aggressive) to high-grade (fast-growing, invasive).

# H&E images


H&E staining is a critical technique in biomedical microscopy for the analysis of tissues. In our images by highlighting key histological features, it aids in distinguishing between low-grade and high-grade brain tumors. Hematoxylin stains nuclei a deep blue purple, emphasizing nuclear size, shape, and chromatin patterns, while eosin stains cytoplasmic and extracellular components pink, providing contrast to tissue architecture.
In low-grade tumors, H&E reveals relatively uniform nuclei, lower cellular density, and minimal mitotic activity. High-grade tumors, in contrast, show marked nuclear atypia, increased mitoses, necrosis, and microvascular proliferation. These features enable pathologists to assess tumor grade and guide therapeutic strategies effectively


# Classifiers


The following machine learning classifiers were evaluated: K
Nearest Neighbors (KNN), Logistic Regression, Support 
Vector Machine (SVM), Radom Forrest and XGBoost. Each 
model was trained in fifty augmented images and evaluated on 
a test set of ten images.

# Data augmentation


Data augmentation techniques had been used to generate  
images, including horizontal and vertical flips, rotations, and 
zoom. These transformations help to reduce overfitting and 
improve the robustness of the models.

# Segmentation

Segmentation is a process used for distinguishing our area of 
interest, with goal the better analysis and extraction of our 
desired features.  
In the current project, Otsu thresholding method has been used 
for segmentation. Otsu algorithm selects a threshold that 
maximizes the intensity between regions, with aim of 
separating the image desired areas.

# Feature extraction and pre-processing 


The images were resized to 96x96 pixels and normalized. 
StandardScaler was used to scale the pixel values to a suitable 
range for machine learning. The images were then flattened into 
1D arrays to serve as input features for the classifiers. 
Additionally, more advanced features were extracted from the 
images to capture detailed texture and structural information. 
These features include some first order features like: Mean, 
standard deviation, skewness, and kurtosis. Additionally, 
second order features are extracted using Gray Level Co
Occurrence Matrix. Some of these are: Contrast, dissimilarity, 
energy, and homogeneity. Also, Local Binary Pattern (LBP) 
method is used. 

# Validation 


For validation of the model, K-fold Cross Validation technique 
is used using GridSearchCV library, which helps to find the best 
hyperparameter of each model. In K-fold validation dataset is 
split into k number of subsets (folds) and be trained to all the 
subsets except one (k-1) subset which is used for the evaluation 
[8]. GridSearchCV function is scikit-learn function that helps 
the tuning of hyperparameters of a machine learning model by 
using different combinations of hyperparameters. In the end of 
the research, the best set is selected based on the metrics.[9]

# Citation

If you use any part of this project in your work, kindly reference it using the following citation:

Fragaki,Z (2024). An Evaluation of Machine Learning Classifiers for Image Classification of Low- and High-Grade Brain Tumor Samples. GitHub. Available at: https://github.com/zeniafragaki/Brain-tumor-discrimination-low-or-high-grade

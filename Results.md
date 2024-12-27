Results:


The results of the evaluation of different models are summarized in Table 1. Logistic Regression performed the best, achieving an accuracy of 83.33% with a dataset of 50 augmented images.


1.Accuracy of different models used: 

![image](https://github.com/user-attachments/assets/ee26383b-5cc7-44c8-a288-2d041ec9253a)

ROC Curve for different Models:

![image](https://github.com/user-attachments/assets/d3265aa1-a9c8-4a4a-a19f-c71a21bbfba5)



4.2. Discussion


The results indicate that Logistic Regression outperforms other classifiers in this case, especially when the dataset size is increased to 50 and 100 images. While KNN provided reasonable results with fewer images, its performance did not significantly improve with the larger dataset. Other more complex models, such as Random Forest and Gradient Boosting, did not outperform Logistic Regression, potentially due to the small dataset.
Data augmentation was crucial in improving model performance. Without augmentation, models were prone to overfitting due to the limited size of the dataset. By applying transformations to the images, the models became more robust and generalized better to unseen data.

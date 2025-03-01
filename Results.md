# Results:


The results of the evaluation of different models are summarized below.


# Accuracy of different models used: 

![image](https://github.com/user-attachments/assets/7e4bbce8-7422-4a6d-8119-536020980beb)

![image](https://github.com/user-attachments/assets/903c7930-0bc2-407b-b25f-5e312a3007d5)

Among the results of  Table  below and Figure above, Random Forest 
algorithm stands out as it had achieved the perfect score of 
100% accuracy to all the test set.Although,cross validation 
score drops to 87% ,could be a sign of not the best 
generalization of the model also could indicate 
overfitting.According the other algorithms SVM and XGBoost 
had achieved slightly good performance with rising scores from 
accuracy to all test set to cross validation accuracy,suggesting 
good performance and generalization of the model.In the other 
hand,combing both results,it shown that Logistic Regression 
and KNN had a very small drop (3.3%) and seems to be a stable 
algorithm with high accuracy scores.

# Cross Validation

![image](https://github.com/user-attachments/assets/72ffa5a6-5b64-448e-8878-70f7e5c67e82)

According to K-Fold Cross-Validation scores of each 
algorithm, it is observed that Random Forest and SVM 
achieved 87% and KNN with Logistic Regression are followed 
by 80% each. Last one with the lower cross validation score, 
but not too low is XGBoost with seventy-three percentage 
(73%).

# ROC Curve

![image](https://github.com/user-attachments/assets/b9086791-5b53-486e-96d5-24b8208bc51f)




# Discussion 


The results, indicate that Random Forrest has the perfect 
accuracy of 100% on the test set. However, in the cross
validation process achieved a lower score, 87%. This result 
shows the ability of the model to identify patterns effectively, 
but the difference might be due to overfitting. Logistic 
Regression and KNN followed with accuracy 83.33%, a quite 
well performance but also it achieved  good cross validation 
score, 80%. On the other hand, SVM and XGBoost show 83.3% 
and 66.7% each, accuracy. Nevertheless, their cross-validation 
accuracy was higher from the accuracy to all the test set ,with 
XGBoost scoring 73% and SVM 87%, respectively. These 
results show that although have low accuracy in the test set, 
their cross-validation scores are higher suggesting a better 
generalization of the model to multiple data splits.


# Conclusion

 
This study evaluates various machine learning algorithms for 
classifying low-grade and high-grade brain tumor images. 
Based on the results, it is observed that Random Forest emerged 
as the best-performing model, achieving an accuracy of 100% 
on the test set but in cross validation we score is significantly 
lower shows overfitting. In contrast ,Logistic Regression scores 
an accuracy of 83.33% and cross validation 80%, indicating a 
signs of overfitting. Also, KNN ,as Logistic Regression had a 
drop to accuracy scores in cross validation. Although, SVM, 
and XGBoost showed higher cross validation scores than test 
set accuracies highlighting their need for further tuning. Data 
augmentation played a significant role in enhancing model 
performance. Future research can focus on improving feature 
extraction methods and applying more advanced deep learning 
techniques, especially for larger datasets. 



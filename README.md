## Toxic Comments Classifier

Identify and classify toxic online comments. Used a dataset from kaggle. 

### Implemented models:
* Logistic regression
* SVM
* LSTM

### Results for Logistic Regression
Class | Precision | Recall | AUC | Lambda
----- | ----- | ----- | ----- | -----
Toxic | 0.88645 | 0.86409 | 0.87527 | 0.00022
Obscene | 0.954 | 0.875 | 0.86547 | 0.00056
Insult | 0.9108 | 0.81649 | 0.76925 | 0.00056

### Tuning Lambda, classification errors
![classification-errors](/traditional_models/imgs/classification-errors-all-horisontal.png)

### Results SVM
Class | Precision | Recall | AUC | Box Constraint | Kernel Scale
----- | ----- | ----- | ----- | ----- | -----
Toxic | 0.87767 | 0.88823 | 0.88568 | 434.90921 | 916.57752
Obscene | 0.94722 | 0.891988 | 0.88043 | 53.3302 | 110.49374
Insult | 0.90742 | 0.82973 | 0.78093 | 34.47461 | 95.08248

### ROC curves and Confusion matrices for both models 
![results-both-models](/traditional_models/imgs/results-both-models.png)

### Results LSTM
Class | Precision | Recall | AUC | F1 Score
----- | ----- | ----- | ----- | -----
Toxic | 0.96368 | 0.83799 | 0.53934 | 0.89645
Severe Toxic | 0.99798 | 0.86144 | 0.50056 | 0.90364
Obscene | 0.99211 | 0.82967 | 0.5123 | 0.90364
Threat | 0.98684 | 0.96026 | 0.52467 | 0.97337
Insult | 0.98 | 0.83088 | 0.51625 | 0.8993
Identity Hate | 0.98613 | 0.93255 | 0.73 | 0.95859

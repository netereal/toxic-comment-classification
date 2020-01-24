clc;
clear;

% read csv file into memmory
filename = "data/train.csv";
opts = detectImportOptions(filename);
all_data = readtable(filename, opts);

% select only 10000 rows for efficiency 
all_data = all_data(1:10000, :);

X = all_data.comment_text;
y = all_data.toxic;

% split data in 70pct train and 30pct test/validation
cv = cvpartition(length(X),'HoldOut',0.3);
X_train = X(cv.training,:);
y_train = y(cv.training,:);
X_test  = X(cv.test,:);
y_test = y(cv.test,:);

% tokenize text with initial cleaning
train_docs = prepare_text(X_train);
test_docs = prepare_text(X_test);

train_bag = bagOfWords(train_docs);
train_bag = removeInfrequentWords(train_bag, 10);

X_train_prepared = tfidf(train_bag);
X_test_prepared = tfidf(train_bag, test_docs);


glm = fitglm(X_train_prepared, y_train, 'linear');

% validate score on test partition
p = predict(glm, X_test_prepared);
[X,Y,T,AUC] = perfcurve(y_test,p, 1);


%SVM
rng(10); % For reproducibility
mdlSVM = fitclinear(X_train_prepared,y_train,...
'Learner','svm',...
'Solver','sparsa','Regularization','lasso',...
'GradientTolerance',1e-8);

score_svm = predict(mdlSVM,X_train_prepared);
[Xsvm,Ysvm,Tsvm,AUCsvm] = perfcurve(y_train,score_svm,'1');

figure
plot(X,Y)
hold on
plot(Xsvm,Ysvm)
xlabel('False positive rate ') 
ylabel('True positive rate')
title('ROC for Classification by Logistic Regression, AUC ' + string(AUC)+ ' and SVM '+ string(AUCsvm))


%% Importing the train dataset 
clc;
clear;

fprintf('Reading data...\n');
opts = detectImportOptions('temp-train-10k.csv');
T = readtable('temp-train-10k.csv',opts);

% Make tfidf - term frequency–inverse document frequency
str = T.comment_text;
documents = tokenizedDocument(str);


bag = bagOfWords(documents);
newBag = removeWords(bag,stopWords);

count = 100;
newBag2 = removeInfrequentWords(newBag,count)

%tbl = topkwords(newBag2,k);


M = tfidf(newBag2);

%full(M(1:10,1:10))

%% Train model - logistic regression (try on 'toxic' field)
x = M;
y = T.toxic;
glm = fitglm(x, y, 'linear', 'distr', 'binomial');


score_log = glm.Fitted.Probability; % Probability estimates

%p = predict(glm, x);       %its the same as p = glm.Fitted.Probability;



%% Receiver operating characteristic (ROC) curve for classifier output
[Xlog,Ylog,Tlog,AUClog] = perfcurve(y,score_log,'1');
AUClog


%SVM
rng(10); % For reproducibility
mdlSVM = fitclinear(x,y);

score_svm = predict(mdlSVM,x);
[Xsvm,Ysvm,Tsvm,AUCsvm] = perfcurve(y,score_svm,'1');
AUCsvm

%% Plot ROC Curves
figure;
plot(Xlog,Ylog)
hold on
plot(Xsvm,Ysvm)
legend('Logistic Regression','SVM','Location','Best')
xlabel('False positive rate'); ylabel('True positive rate');
title('ROC Curves for Logistic Regression and SVM Classification')
hold off





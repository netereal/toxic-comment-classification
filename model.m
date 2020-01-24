% Importing the train dataset 
clc;
clear;

fprintf('Reading data...\n');
opts = detectImportOptions('temp-train-10k.csv');
T = readtable('temp-train-10k.csv',opts);

% Make tfidf - term frequency–inverse document frequency
str = T.comment_text;
documents = tokenizedDocument(str);

k = 5000;
bag = bagOfWords(documents);
newBag = removeWords(bag,stopWords);

count = 1000;
newBag2 = removeInfrequentWords(newBag,count)

%tbl = topkwords(newBag2,k);


M = tfidf(newBag2);

%full(M(1:10,1:10))

%Train model - logistic regression (try on 'toxic' field)
x = M;
y = T.toxic;
glm = fitglm(x, y, 'linear', 'distr', 'binomial');


score_log = glm.Fitted.Probability; % Probability estimates

%p = predict(glm, x);       %its the same as p = glm.Fitted.Probability;



%Receiver operating characteristic (ROC) curve or other performance curve for classifier output
[Xlog,Ylog,Tlog,AUClog] = perfcurve(y,score_log,'1');
AUClog



mdlNB = fitcnb(x,y);
score_nb = resubPredict(mdlNB);
[Xnb,Ynb,Tnb,AUCnb] = perfcurve(y,score_nb,'1');
AUCnb



figure;
plot(Xlog,Ylog)
hold on
plot(Xnb,Ynb)
legend('Logistic Regression','Naive Bayes','Location','Best')
xlabel('False positive rate'); ylabel('True positive rate');
title('ROC Curves for Logistic Regression, SVM, and Naive Bayes Classification')
hold off





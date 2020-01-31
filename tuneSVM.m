clc;
clear;

% read csv file into memmory
filename = "data/train.csv";
opts = detectImportOptions(filename);
all_data = readtable(filename, opts);

% select only 10000 rows for efficiency 
all_data = all_data(1:30000, :);

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



%Logistic regression

Lambda = logspace(-10,-0.5,11);


rng(10); % For reproducibility
CVMdl = fitclinear(X_train_prepared',y_train,'ObservationsIn','columns','KFold',5,...
    'Learner','svm','Solver','sparsa','Regularization','lasso',...
    'Lambda',Lambda,'GradientTolerance',1e-8)

numCLModels = numel(CVMdl.Trained)

Mdl1 = CVMdl.Trained{1}

ce = kfoldLoss(CVMdl);


Mdl = fitclinear(X_train_prepared',y_train,'ObservationsIn','columns',...
    'Learner','svm','Solver','sparsa','Regularization','lasso',...
    'Lambda',Lambda,'GradientTolerance',1e-8);
numNZCoeff = sum(Mdl.Beta~=0);


figure;
subplot(1,2,1);
hL1 = plot(log10(Lambda),log10(ce)); 
hL1.Marker = 'o';
ylabel('log_{10} classification error')
xlabel('log_{10} Lambda')
title('Test-Sample Statistics')
hold off

idxFinal = 2;

MdlFinal = selectModels(Mdl,idxFinal);

% show score on test partition
p = predict(MdlFinal, X_test_prepared);
[X1,Y1,T1,AUC1] = perfcurve(y_test,p, 1);


subplot(1,2,2);
plot(X1,Y1)

legend('Model number '+string(idxFinal))
xlabel('False positive rate') 
ylabel('True positive rate')
title('Find Good Lasso Penalty Using Cross-Validation,  AUC = ' + string(AUC1))




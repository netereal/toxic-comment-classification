%{
To implement this model we use ‘fitclinear’ function
from Statistics and Machine Learning Toolbox. As a learner we specify
Logistic Regression with ”lasso” regularization type.We also specify 
objective function minimization technique as SpaRSA (Sparse Reconstruction 
by Separable Approximation), Lower the tolerance on the gradient of the 
objective function to 10^?8 and set up a regularization term Lambda.

We get the optimal value of this hyper-parameter after manual
tuning using 'tuneLogisticRegression'.
%}
clc;
clear;

%% read csv file into memmory
filename = "../data/train.csv";
opts = detectImportOptions(filename);
all_data = readtable(filename, opts);

%% resample data
% you can comment this line to see how model performs without resampling
all_data = resample(all_data);
% all_data = all_data(1:10000, :);

% results table
results = table([0;0;0],[0;0;0],[0;0;0],[0;0;0],...
          'VariableNames',{'precision','recall','AUC','Lambda'},...
          'RowNames',{'toxic','obscene','insult'});
      
%% for each class (toxic, obese, insult)
for i = 1:3
    X = all_data.comment_text;
    % take one class
    class = all_data.Properties.VariableNames(:,i+2);
    y = all_data{:,string(class)};

    %% split data in 70pct train and 30pct test
    cv = cvpartition(length(X),'HoldOut',0.3);
    X_train = X(cv.training,:);
    y_train = y(cv.training,:);
    X_test  = X(cv.test,:);
    y_test = y(cv.test,:);

    %% tokenize text with initial cleaning
    train_docs = prepare_text(X_train);
    test_docs = prepare_text(X_test);

    train_bag = bagOfWords(train_docs);
    train_bag = removeInfrequentWords(train_bag, 10);

    X_train_prepared = tfidf(train_bag);
    X_train_prepared = full(X_train_prepared);
    X_test_prepared = tfidf(train_bag, test_docs);

    %% logistic regression
    % Find best lambda
    Lambda = tuneLogisticRegression(X_train_prepared, y_train);


    glm = fitclinear(X_train_prepared',y_train,'ObservationsIn','columns',...
        'Learner','logistic','Solver','sparsa','Regularization','lasso',...
        'Lambda',lambda,'GradientTolerance',1e-8);
    %% validate score on test partition
    p = predict(glm, X_test_prepared);

    %% Plot ROC curve and Confusion matrix
    figure;
    subplot(1,2,1);
    [X,Y,T,AUC] = perfcurve(y_test,p, 1);
    plot(X,Y)
    legend('Logistic Regression for ' + string(class))
    xlabel('False positive rate ') 
    ylabel('True positive rate')
    title('ROC for Logistic Regression, AUC='+ string(AUC))
    subplot(1,2,2);
    [cm, order] = confusionmat(categorical(y_test), categorical(p), 'Order', {'0', '1'});
        confusionchart(cm, order);
        title('Confusion matrix for ' + string(class))
        TP = cm(1,1);
        FP = cm(1,2);
        FN = cm(2,1);
        recall = TP/(TP+FN);
        precision = TP/(TP+FP);
        
    results(i,:).precision = precision;
    results(i,:).recall = recall;
    results(i,:).AUC = AUC;
    results(i,:).Lambda = Lambda;

end
disp(results);
writetable(results,'resultsLR.csv');
type 'resultsLR.csv';

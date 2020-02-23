%{
We implement 5-fold cross-validation to find
optimal parameter Lambda for Logistic Regression. We have
5 folds: 4 for training and 1 is test fold. These steps were
implemented:

- Create a set of 15 logarithmically-spaced regularization strengths from 
10^?6 through 10^?0.5, which are used to create a series of different models.
- Cross-validate the models.
- Estimate the cross-validated classification error.
- Choose the index of lambda that has low classification error.

We choose lambda by looking at the classification error rates and finding 
the one with smallest prediction error. Sometimes taking the previous to 
our ”optimal” lambda produce slightly better results, so we decided to 
choose this one as an optimal.
%}
function lambda = tuneLogisticRegression(X_train_prepared, y_train)

    %Logistic regression lambda
    Lambda = logspace(-6,-0.5,15);

    rng(10); % For reproducibility
    CVMdl = fitclinear(X_train_prepared',y_train,'ObservationsIn','columns','KFold',5,...
        'Learner','logistic','Solver','sparsa','Regularization','lasso',...
        'Lambda',Lambda,'GradientTolerance',1e-8);

    ce = kfoldLoss(CVMdl);

%     % Plot Classification Errors
%     figure;
%     hL1 = plot(log10(Lambda),log10(ce)); 
%     hL1.Marker = 'o';
%     ylabel('log_{10} classification error')
%     xlabel('log_{10} Lambda')
%     title('Classification Error')
%     hold off
    
    % find model with lowest classification error
    [min_vals, min_idx] = min(ce);
    % select model previous to this one
    idxFinal = min_idx - 1;

    lambda = Lambda(:,idxFinal);
end

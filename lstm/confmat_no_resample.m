classes = ["Toxic", "Severe Toxic", "Obscene", "Threat", "Insult", "Identity Hate"];

for col = 3 : length(data.Properties.VariableNames)
    data.(col) = categorical(data.(col));
    cvp = cvpartition(data.(col),'Holdout',0.3);
    dataTrain = data(training(cvp),:);
    dataHeldOut = data(test(cvp),:);
    cvp = cvpartition(dataHeldOut.(col),'Holdout',0.5);
    dataValidation = dataHeldOut(training(cvp), :);
    dataTest = dataHeldOut(test(cvp), :);
    YTest = dataTest.(col);
    net = networks(col-2);
    YPred = classify(net,XTest);
    [cm, order] = confusionmat(YTest,YPred, 'Order', {'0', '1'});
    %confusionchart(cm, order);
    TP = cm(1,1);
    FP = cm(1,2);
    FN = cm(2,1);
    recall = TP/(TP+FN);
    precision = TP/(TP+FP);
    F1Score = (2*precision*recall) / (precision+recall);
    fprintf("Class %s, F1 Score: %.5f\n", classes(col-2), F1Score);
    %fprintf("Press any key to continue\n");
    %pause;
end
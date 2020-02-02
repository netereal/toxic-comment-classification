classes = ["Toxic", "Severe Toxic", "Obscene", "Threat", "Insult", "Identity Hate"];

for col = 3 : length(data.Properties.VariableNames)
    data.(col) = categorical(data.(col));
    positive = table(data.comment_text(data.(col) == categorical(1)), ...
                data.(col)(data.(col) == categorical(1)));
    if(height(positive) > 2000)
        positive = positive(1:2000, :);
    end
    negative = table(data.comment_text(data.(col) == categorical(0)), ...
                data.(col)(data.(col) == categorical(0)));
    negative = negative(1:11500 - height(positive), :);
    subData = [negative; positive];
    cvp = cvpartition(subData.Var2,'Holdout',0.2);
    dataTrain = subData(training(cvp),:);
    dataTest = subData(test(cvp),:);
    YTest = dataTest.Var2;
    net = networks(col-2);
    YPred = classify(net,XTest);
    [cm, order] = confusionmat(YTest,YPred, 'Order', {'0', '1'});
    confusionchart(cm, order);
    TP = cm(1,1);
    FP = cm(1,2);
    FN = cm(2,1);
    recall = TP/(TP+FN);
    precision = TP/(TP+FP);
    F1Score = (2*precision*recall) / (precision+recall);
    fprintf("Class %s, F1 Score: %.5f\nPress any key to continue\n", classes(col-2), F1Score);
    pause;
end
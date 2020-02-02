%% Initialization
clc;
clear;
%% CONSTANTS
train_filename = "train.csv";
%% Read Data
fprintf('Reading data...\n');
opts = detectImportOptions(train_filename);
data = readtable(train_filename,opts);
%% Preprocess Data
idxEmpty = strlength(data.comment_text) == 0;
data(idxEmpty,:) = [];
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
    %% TRAIN FOR CURRENT CLASS
    % Extract the text data and labels from the partitioned tables
    fprintf('Extract the text data and labels from the partitioned tables...\n');
    textDataTrain = dataTrain.Var1;
    textDataTest = dataTest.Var1;
    YTrain = dataTrain.Var2;
    YTest = dataTest.Var2;
    % Get prepared documents from text data
    documentsTrain = prepare_text(textDataTrain);
    % Convert Documents to Sequences
    enc = wordEncoding(documentsTrain);
    XTrain = doc2sequence(enc,documentsTrain);
    disp('Configuring training labels and options...');
    % Train the model
    inputSize = 1;
    embeddingDimension = 100;
    numWords = enc.NumWords;
    numHiddenUnits = 180;
    numClasses = numel(categories(YTrain));

    layers = [ ...
        sequenceInputLayer(inputSize)
        wordEmbeddingLayer(embeddingDimension,numWords)
        lstmLayer(numHiddenUnits,'OutputMode','last')
        fullyConnectedLayer(numClasses)
        softmaxLayer
        classificationLayer];

    options = trainingOptions('adam', 'MaxEpochs',10, ...    
        'GradientThreshold',1, ...
        'InitialLearnRate',0.01, ...
        'Plots','training-progress', ...
        'ExecutionEnvironment', 'gpu', ...
        'MiniBatchSize', 32, ...
        'Verbose',false);

    disp('Starting training...');
    net = trainNetwork(XTrain,YTrain,layers,options);
    %% Testing Documents
    disp('Preparing Testing Enviroment...');
    documentsTest = prepare_text(textDataTest);
    XTest = doc2sequence(enc,documentsTest);
    YPred = classify(net,XTest);
    networks(col-2) = net;
    accuracy(col-2) = sum(YPred == YTest)/numel(YPred);
end
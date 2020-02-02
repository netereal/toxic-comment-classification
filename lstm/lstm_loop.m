%% Initialization
clc;
clear;
%% CONSTANTS
train_filename = "train.csv";
%% Read Data
fprintf('Reading data...\n');
opts = detectImportOptions(train_filename);
data = readtable(train_filename,opts);
% Training data will be around 10000
% Testing data will be around 2500
data = data(1: 15500, :);
%% Preprocess Data
idxEmpty = strlength(data.comment_text) == 0;
data(idxEmpty,:) = [];
for col = 3 : length(data.Properties.VariableNames)
    %% Prepare Data
    data.(col) = categorical(data.(col));
    cvp = cvpartition(data.(col),'Holdout',0.3);
    dataTrain = data(training(cvp),:);
    dataHeldOut = data(test(cvp),:);
    cvp = cvpartition(dataHeldOut.(col),'HoldOut',0.5);
    dataValidation = dataHeldOut(training(cvp),:);
    dataTest = dataHeldOut(test(cvp),:);
    %% TRAIN FOR CURRENT CLASS
    % Extract the text data and labels from the partitioned tables
    fprintf('Extract the text data and labels from the partitioned tables...\n');
    textDataTrain = dataTrain.comment_text;
    textDataTest = dataTest.comment_text;
    textDataValidation = dataValidation.comment_text;
    YTrain = dataTrain.(col);
    YValidation = dataValidation.(col);
    YTest = dataTest.(col);
    % Get prepared documents from text data
    documentsTrain = prepare_text(textDataTrain);
    documentsValidation = prepare_text(textDataValidation);
    % Convert Documents to Sequences
    enc = wordEncoding(documentsTrain);
    XTrain = doc2sequence(enc,documentsTrain);
    XValidation = doc2sequence(enc, documentsValidation);
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
        'ValidationData', {XValidation, YValidation}, ...
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
%% Result
for i = 3 : length(data.Properties.VariableNames)
    classname = data.Properties.VariableNames(i);
    fprintf("For class name %s, accuracy is %.3f\n", classname, accuracy(i-2));
end
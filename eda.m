
% read csv file into memmory
filename = "data/train.csv";
opts = detectImportOptions(filename);
all_data = readtable(filename, opts);

% select only 10000 rows for efficiency 
all_data = all_data(1:10000, :);
comments = all_data.comment_text;

rawDocuments = tokenizedDocument(comments);
rawBag = bagOfWords(rawDocuments)

% preprocess and clean text
comments = prepare_text(comments);

cleanDocuments = tokenizedDocument(comments);
cleanedBag = bagOfWords(cleanDocuments)

%Compare the raw data and the cleaned data by visualizing the two bag-of-words models using word clouds.
figure
subplot(1,2,1)
wordcloud(rawBag);
title("Raw Data")
subplot(1,2,2)
wordcloud(cleanedBag);
title("Cleaned Data")

% Sentence Length Distribution


% Word cloud for the most common words
figure
wordcloud(comments);
title("Word cloud on all document")







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

% cleanDocuments = tokenizedDocument(comments);
% cleanedBag = bagOfWords(cleanDocuments)

%% Compare the raw data and the cleaned data by visualizing the two bag-of-words models using word clouds.
% figure
% subplot(1,2,1)
% wordcloud(rawBag);
% title("Raw Data")
% subplot(1,2,2)
% wordcloud(cleanedBag);
% title("Cleaned Data")

%% Sentence Length Distribution
% figure
% sentenceLength = doclength(comments);
% histogram(sentenceLength,50)
% title("Sentence Length Distribution")
% xlabel('Number of words in comment') 
% ylabel('Count of comments')

%% Word cloud for the most common words
% figure
% subplot(1,2,1)
% wordcloud(comments);
% title("Word cloud on all document")
% % Top Ten most Common Words
% subplot(1,2,2)
% top10 = topkwords(cleanedBag, 10);
% X_top10 = categorical(top10.Word);
% X_top10 = reordercats(X_top10,top10.Word);
% Y_top10 = top10.Count;
% bar(X_top10,Y_top10)
% title("Top Ten most Common Words")

%% Split the data into the classes
toxic = all_data(all_data.toxic==1, 2);
severe_toxic = all_data(all_data.severe_toxic==1, 2);
obscene = all_data(all_data.obscene==1, 2);
threat = all_data(all_data.threat==1, 2);
insult = all_data(all_data.insult==1, 2);
identity_hate = all_data(all_data.identity_hate==1, 2);

%% Create class distribution histogram 
% class_names = ["toxic","severe toxic","obscene","threat","insult","identity hate"]';
% class_count = [height(toxic),height(severe_toxic),height(obscene),height(threat),height(insult),height(identity_hate)]';
% 
% class_distribution = table(class_names,class_count,'VariableNames', {'class_names','class_count'});
% 
% figure
% X_class = categorical(class_distribution.class_names);
% X_class = reordercats(X_class,class_distribution.class_names);
% bar(X_class,class_distribution.class_count)
% title("Class Distribution")

%% Word clouds for all classes




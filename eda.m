
% read csv file into memmory
filename = "data/train.csv";
opts = detectImportOptions(filename);
all_data = readtable(filename, opts);

% select only 10000 rows for efficiency 
all_data = all_data(1:10000, :);
comments = all_data.comment_text;

rawDocuments = tokenizedDocument(comments);
rawBag = bagOfWords(rawDocuments);

% preprocess and clean text

all_data.comment_text = prepare_text(all_data.comment_text);

% cleanDocuments = tokenizedDocument(all_data.comment_text);
% cleanedBag = bagOfWords(cleanDocuments)

%% Compare the raw data and the cleaned data using word clouds.
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
% toxic = all_data(all_data.toxic==1, 2);
% severe_toxic = all_data(all_data.severe_toxic==1, 2);
% obscene = all_data(all_data.obscene==1, 2);
% threat = all_data(all_data.threat==1, 2);
% insult = all_data(all_data.insult==1, 2);
% identity_hate = all_data(all_data.identity_hate==1, 2);
% clean = all_data(all_data.identity_hate==0 & all_data.insult==0 & all_data.threat==0 & all_data.obscene==0 & all_data.severe_toxic==0 & all_data.toxic==0, 2);

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

%% Check for Multiple tagging
% multi_names = categorical({'clean','tagged','all'});
% multi_names = reordercats(multi_names,{'tagged','clean','all'});
% sum_tags = height(toxic)+height(severe_toxic)+height(obscene)+height(threat)+height(insult)+height(identity_hate);
% multi_count = [height(clean) sum_tags height(all_data)];
% 
% figure
% bar(multi_names,multi_count)
% title("Clean: " + height(clean) + ", Tagged: " + sum_tags + ", All: " + height(all_data));

%% Word clouds for all classes
% toxicBag = bagOfWords(tokenizedDocument(toxic.comment_text));
% figure
% subplot(2,3,1)
% wordcloud(toxicBag);
% title("Toxic")
% 
% severeToxicBag = bagOfWords(tokenizedDocument(severe_toxic.comment_text));
% subplot(2,3,2)
% wordcloud(severeToxicBag);
% title("Severe Toxic")
% 
% obsceneBag = bagOfWords(tokenizedDocument(obscene.comment_text));
% subplot(2,3,3)
% wordcloud(obsceneBag);
% title("Obscene")
% 
% threatBag = bagOfWords(tokenizedDocument(threat.comment_text));
% subplot(2,3,4)
% wordcloud(threatBag);
% title("Threat")
% 
% insultBag = bagOfWords(tokenizedDocument(insult.comment_text));
% subplot(2,3,5)
% wordcloud(insultBag);
% title("Insult")
% 
% identityHateBag = bagOfWords(tokenizedDocument(identity_hate.comment_text));
% subplot(2,3,6)
% wordcloud(identityHateBag);
% title("Identity hate")

%% Create Word Clouds of Bigrams
% bag = bagOfNgrams(tokenizedDocument(all_data.comment_text));

% mdl = fitlda(bag,10);
% 
% figure
% for i = 1:6
%     subplot(2,3,i)
%     wordcloud(mdl,i);
%     title("LDA Topic " + i)
% end

%% Top Ten most Common Bigrams
% top10ngram = topkngrams(bag, 10);
% X_ngram10 = categorical(top10ngram.Ngram(:,1) +' '+ top10ngram.Ngram(:,2));
% X_ngram10 = reordercats(X_ngram10,top10ngram.Ngram(:,1) +' '+ top10ngram.Ngram(:,2));
% Y_ngram10 = top10ngram.Count;
% bar(X_ngram10,Y_ngram10)
% title("Top Ten most Common Bigrams")







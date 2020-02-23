    %{
    Function to preprocess text.
    These steps involved were:
        - converting all the text data into lower case
        - tokenizing the text (breaking down each sentence by
        splitting into words)
        - erasing the punctuation
        - removing stop words such as ”and”, ”but”, ”the” etc,
        which don’t influence the meaning of a sentence drastically
        - removing words with two or fewer characters like ”of”,
        ”to” etc
        - removing words with fifteen or more characters
        - lemmatizing the word corpus (converting word stems like
        tenses, infinitives, etc into their original word forms)
    %}
function documents = prepare_text(textData)
    % Convert the text data to lowercase.
    documents = lower(textData);
    % Tokenize the text.
    documents = tokenizedDocument(documents);

    % Remove a list of stop words then lemmatize the words. To improve
    % lemmatization, first use addPartOfSpeechDetails.
    documents = addPartOfSpeechDetails(documents);
    % Remove a list of stop words.
    documents = removeStopWords(documents);
    documents = normalizeWords(documents,'Style','lemma');

    % Erase punctuation.
    documents = erasePunctuation(documents);

    % Remove words with 2 or fewer characters, and words with 15 or more
    % characters.
    documents = removeShortWords(documents,2);
    documents = removeLongWords(documents,15);
end


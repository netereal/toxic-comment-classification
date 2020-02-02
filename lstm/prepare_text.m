function documents = prepare_text(textData)
    % Tokenize the text.
    documents = tokenizedDocument(textData);
    % Convert to lowercase.
    documents = lower(documents);
    % Remove stop words
    documents = removeStopWords(documents);
    % Erase punctuation.
    documents = erasePunctuation(documents);

    % Remove words with 15 or more characters.
    documents = removeLongWords(documents,15);
    
    % Remove a list of stop words then lemmatize the words. To improve
    % lemmatization, first use addPartOfSpeechDetails.
    documents = addPartOfSpeechDetails(documents);
    documents = normalizeWords(documents,'Style','lemma');
end


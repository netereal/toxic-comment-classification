function resultsTable = results(enc,networks)
    %comms = ["This is nice", "GO TO HELL SHITFACE", "Keep it up",...
    %         "Shut up Karen", "I hope you get cancer", "I will kill you", ...
    %         "I will kill you", "Giant midget", "Man baby", "Bullshit, lol", "The fuck is wrong with you?",...
    %         "Your fingers look like toes","Even your reflection should bully you","Why the wide face?","The girl next dork", ...
    %         "Botox John Travolta with full blown AIDS.","The furniture has a brighter future than you.",...
    %         "You can tell he?s Scottish because he?ll never be independent", "Nah, this is straight up Mick Foley before he fell off Hell in a Cell",...
    %         "Your arms look 20 years older than your face", "Kinda weird to spend $40k+ to get a McDonald's diploma.",...
    %         "You look like John Lennon. After getting shot.", "Is this a roastme pic or a mugshot?", "Mohammad Jackson.",...
    %         "You must be rich.","And let's give a hand to... oh, sorry."];
    comms = ["YOU RAT BASTARD, SON OF A BITCH.. JESUS!!?!?!?!!", "Must you always be so cheerful? You dumb, empty headed bimbo", "Her mother was a slut too.", "Eat dirt and die.. you trashy hoe", "Haha.. tHe jOkEs On Yu..", "I AM JUST MAAAD.. MAD I TELL YOU!!!", "have you no brain????", "This is nice", "GO TO HELL SHITFACE", "Keep it up", "Shut up Karen.", "I hope you get cancer", "I will kill you" , "Bullshit, lol", "The fuck is wrong with you?", "Your fingers look like toes", "Even your reflection should bully you", "Botox John Travolta with full blown AIDS.", "The furniture has a brighter future than you.", "You can tell he?s Scottish because he?ll never be independent", "Your arms look 20 years older than your stupid face", "Kinda weird to spend $40k+ to get a McDonald's diploma.", "You look like John Lennon. After getting shot."];
    docs = prepare_text(comms);
    test = doc2sequence(enc, docs);
    resultsTable = table(comms');
    for i = 1 : length(networks)
        pred = classify(networks(i), test);
        resultsTable = addvars(resultsTable, pred);
    end
    resultsTable.Properties.VariableNames = {'Comment' 'Toxic' 'Severe_Toxic' 'Obscene' 'Threat' 'Insult' 'Identity_Hate'};
end
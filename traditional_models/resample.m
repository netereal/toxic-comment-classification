%{
This code takes an original data with 170k comments, resamples it and 
returns for training.

Number of comments of each class in train file:
Total - 178394
Clean - 143300
Toxic - 15290
Severe toxic - 1595
Obscene - 8449
Threat - 478
Insult - 7877
Identity hate – 1405

We take 3 classes that has the biggest numbers of comments. Here is the
resulting proportion:
Clean - 10000
Toxic - 11329
Obscene - 8449
Insult - 7877
All - 22184
%}
function resampled = resample(all_data)
all_data= removevars(all_data,{'identity_hate','threat','severe_toxic'});

% Split the data into the classes
toxic = all_data(all_data.toxic==1, :);
obscene = all_data(all_data.obscene==1, :);
insult = all_data(all_data.insult==1, :);
clean = all_data(all_data.insult==0 & all_data.obscene==0 & all_data.toxic==0, :);

resampled = vertcat(toxic(1:5000, :), obscene, insult, clean(1:10000, :));
% remove duplicate rows
resampled = unique(resampled);
% shuffle rows randomly
resampled=resampled(randsample(1:height(resampled),height(resampled)),:);
end



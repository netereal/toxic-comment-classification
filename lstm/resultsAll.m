clear;
load all6net_adam;
resultFinal = results(enc, networks);
clearvars -except resultFinal;
load all6net_adam_resampled;
resultTemp = results(enc, networks);
for i = 1 : length(networks)
    row = resultFinal.(i+1);
    tempRow = resultTemp.(i+1);
    if(i ~= 1)
        for j = 1 : length(row)
            if(row(j) == categorical(0))
                row(j) = tempRow(j);
            end
        end
    end
    
    resultFinal.(i+1) = row;
end
resultFinal
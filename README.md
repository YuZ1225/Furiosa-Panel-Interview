# Furiosa Panel Interview
## Menu
* [Resume](https://github.com/YuZ1225/Furiosa-Panel-Interview/blob/master/resume/resume_yuzhao.pdf)
* [Projects](#project)
* [Code Snippet](#code)

```Matlab
%% Naive Bayes Collaborative Filter

function [NB_output]= NB(data_matrix, rating_to_pred)
fprintf('The total running time is about 720s. \n\n');
[label_num, ~] = size(rating_to_pred);

% pred_label = zeros(label_num, 1);
tmp_prob = zeros(label_num,5);


for i = 1:1:label_num
    tic;
    tmp_pred_line = data_matrix(rating_to_pred(i, 2), :);
    pred_norzero_num = numel(find(tmp_pred_line ~= 0));
    tmp_numofbook = sum(data_matrix(:, rating_to_pred(i, 1)) >0);
    for j = 1:5
        tmp_prior_num = sum(data_matrix(:, rating_to_pred(i, 1)) == j);
        if(tmp_prior_num == 0)
            tmp_prob(i, j) = -Inf;
        else
            tmp_book = find(data_matrix(:, rating_to_pred(i, 1)) == j);
            tmp_test = data_matrix(tmp_book, :);
            tmp_posterior = sum((tmp_test ~= 0 & tmp_test == tmp_pred_line), 1);
            tmp_notzero = find(tmp_posterior ~= 0);
            tmp_same = full(tmp_posterior(tmp_notzero));
            tmp_result = (tmp_same + 1)/(tmp_prior_num + 5);
            tmp_result = sum(log(tmp_result)) + log(1/(tmp_prior_num + 5))*(pred_norzero_num - numel(tmp_notzero)) + log(tmp_prior_num/numel(tmp_numofbook));
            tmp_prob(i, j) = tmp_result;       
        end
    end
    fprintf('number %d\n', i);
    toc;
end

[~, pred_label] = max(tmp_prob, [], 2);
true_label = rating_to_pred(:, 3);
% CCR = numel(find(pred_label == true_label))/numel(true_label);
MAE = mean(abs(pred_label - true_label));
RMSE = sqrt(immse(pred_label, true_label));
NB_output.MAE = MAE;
NB_output.RMSE = RMSE;
NB_output.pred_label = pred_label;

end
```

[Back to Menu](#menu)
<br>

[Back to Menu](#menu)
<br>

[Back to Menu](#menu)
<br>

[Back to Menu](#menu)
<br>

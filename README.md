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
<br>


```Matlab
%% Probabolistic Matrix Factorization

function [PMF_output]= PMF(data_matrix, rating_to_pred, iteration, d, weight_missing, mu, lambda, rm)
% Input: sparse data matrix that need to be factorized, with size m x n
%        matrix of rating to pred with user_id, item_id and rating
% Output: PMF_out include all the output of PMF
%         PMF.MAE; PMF.RMSE
%         PMF.U: item martix with m x d
%         PMF.V: user matrix with n x d
%         PMF_output.U = U;
%         PMF_output.V = V;
%         PMF_output.MAE_iter = MAE_iter;
%         PMF_output.RMSE_iter = RMSE_iter;
%         PMF_output.MAE_iter_round = MAE_iter_round;
%         PMF_output.RMSE_iter_round = RMSE_iter_round;
%         PMF_output.label_pred = label_pred;
%         PMF_output.label_pred_round = label_pred_round;
%         PMF_output.MAE_baseline = MAE_baseline;
%         PMF_output.RMSE_baseline = RMSE_baseline;
  

    [itm_num, usr_num] = size(data_matrix);
%     iter = 100;          % iteraton number
%     
%     para_d = 10;    % dimension of latent feature
%     para_weight = 0.05;    % parameter of weight
%     para_mu = 0.005;    % leaning rate
%     para_lambda = 0.2;  % regularization parameter
%     para_rm = mean(data_matrix(data_matrix>0)); % offset of missing rating
%     para_rm = 0;
    iter = iteration;
    para_d = d;
    para_weight = weight_missing;
    para_mu = mu;
    para_lambda = lambda;
    para_rm = rm;
    
    baseline = round(mean(data_matrix(data_matrix>0)));
    MAE_baseline = mean(abs(rating_to_pred(:,3) - baseline));
    RMSE_baseline = sqrt(mean((rating_to_pred(:,3) - baseline).^2));
    
%     bad initialize    
%     U = ones(itm_num, para_d); % item matrix
%     V = ones(usr_num, para_d); % user matrix
%     U = U / sqrt(para_d/5);
%     V = V / sqrt(para_d);

    U = (1 * randn(para_d, itm_num) + 1/sqrt(para_d/3))';
    V = (1 * randn(para_d, usr_num) + 1/sqrt(para_d/3))';
    
    W = double(full(data_matrix>0));
    
    W = sparse(W);  % for the entire data set
%     tmp_W(W == 0) = para_weight;    % weight matrix
%     W_m = double(W == 0);
    MAE_iter = zeros(iter, 1);
    RMSE_iter = zeros(iter, 1);
    MAE_iter_round = zeros(iter, 1);
    RMSE_iter_round = zeros(iter, 1);
    
    [label_num, ~] = size(rating_to_pred);
    label_pred = zeros(label_num, iter);
    label_pred_round = zeros(label_num, iter);
    
    tmp_index = (rating_to_pred(:,1)-1) * itm_num + rating_to_pred(:,2);
    
    for i = 1:1:iter
        tic;
%         for entire data set this is too slow
%         U = U - para_mu * (-(tmp_W .* (data_matrix - (para_rm + U*V'))) * V + para_lambda * U);
%         V = V - para_mu * (-(tmp_W .* (data_matrix - (para_rm + U*V')))' * U + para_lambda * V);

        U = U - para_mu * (-(W .* (data_matrix - U*V')) * V + para_lambda * U);
        V = V - para_mu * (-(W .* (data_matrix - U*V'))' * U + para_lambda * V);

%         incorrect method. these parameter cannot be optimized by this way
%         para_lambda = para_lambda - para_mu/100*(sum(sum(U.^2)) + sum(sum(V.^2)))/2;
%         para_weight = para_weight - para_mu/10000* sum(sum(((W_m .* (data_matrix - (para_rm + U*V'))).^2)))/2;
%         para_rm = para_rm - para_mu/100*sum(sum(-(tmp_W .* (data_matrix - (para_rm + U*V')))));
        
        tmp_result = (U*V');
        label_pred(:, i) = tmp_result(tmp_index);
        MAE_iter(i, 1) = mean(abs(label_pred(:, i) - rating_to_pred(:,3)));
        RMSE_iter(i, 1) = sqrt(immse(label_pred(:, i), rating_to_pred(:,3)));
        
%         tmp_result = round(U*V');
        label_pred_round(:, i) = round(label_pred(:, i));
        MAE_iter_round(i, 1) = mean(abs(label_pred_round(:, i) - rating_to_pred(:,3)));
        RMSE_iter_round(i, 1) = sqrt(immse(label_pred_round(:, i), rating_to_pred(:,3)));
        tmp_result = [];    % clear memory
        toc;
    end
    
    % save tmp.mat U V label_pred MAE_iter RMSE_iter label_pred_round
    % MAE_iter_round RMSE_iter_round i para_d para_weight para_mu
    % para_lambda data_matrix rating_to_pred W tmp_index MAE_baseline RMSE_baseline
    
    % PMF_output_d10 = PMF_output
    % save PMF_entire_d10.mat PMF_output_d10
    % PMF_output_d3 = PMF_output
    % save PMF_entire_d3.mat PMF_output_d3
    
    PMF_output.U = U;
    PMF_output.V = V;
    PMF_output.MAE_iter = MAE_iter;
    PMF_output.RMSE_iter = RMSE_iter;
    PMF_output.MAE_iter_round = MAE_iter_round;
    PMF_output.RMSE_iter_round = RMSE_iter_round;
    PMF_output.label_pred = label_pred;
    PMF_output.label_pred_round = label_pred_round;
    PMF_output.MAE_baseline = MAE_baseline;
    PMF_output.RMSE_baseline = RMSE_baseline;
end
```
<br>

```Matlab
%% test and optimize parameter of PMF
% [PMF_output]= PMF(data_matrix, rating_to_pred, iteration, d, weight_missing, mu, lambda, rm)
%         PMF_output.U = U;
%         PMF_output.V = V;
%         PMF_output.MAE_iter = MAE_iter;
%         PMF_output.RMSE_iter = RMSE_iter;
%         PMF_output.MAE_iter_round = MAE_iter_round;
%         PMF_output.RMSE_iter_round = RMSE_iter_round;
%         PMF_output.label_pred = label_pred;
%         PMF_output.label_pred_round = label_pred_round;
%         PMF_output.MAE_baseline = MAE_baseline;
%         PMF_output.RMSE_baseline = RMSE_baseline;
%% optimize d;
load('data.mat')
data_matrix = cellofmatrix{5};
[itm_num, usr_num] = size(data_matrix);

iteration = 100;
weight_missing = 0;
mu = 0.005;
lambda = 0.01;
rm = 0;
d = 20;

tmp_rmse = zeros(d,1);
tmp_rmse_round = zeros(d,1);
tmp_mae = zeros(d,1);
tmp_mae_round = zeros(d,1);
for i = 1:1:d
    [PMF_output] = PMF(data_matrix, rating_to_pred, iteration, i, weight_missing, mu, lambda, rm);
    tmp_rmse(i, 1) = PMF_output.RMSE_iter(iteration);
    tmp_rmse_round(i, 1) = PMF_output.RMSE_iter_round(iteration);
    tmp_mae(i, 1) = PMF_output.MAE_iter(iteration);
    tmp_mae_round(i, 1) = PMF_output.MAE_iter_round(iteration);
end
j = 1:1:d;
y = sin(j) - sin(j) + PMF_output.RMSE_baseline;
figure;
subplot(1,2,1)
% plot(j, tmp_rmse, '*-');
% hold on
plot(j, tmp_rmse_round, '*-');
hold on
% line([1,d], [PMF_output.RMSE_baseline, PMF_output.RMSE_baseline])
plot(j, y)
hold off
legend('pmf', 'baseline')
title('RMSE VS Different d')
xlabel('d')
ylabel('RMSE')

y = sin(j) - sin(j) + PMF_output.MAE_baseline;
subplot(1,2,2)
% plot(j, tmp_mae, '*-');
% hold on
plot(j, tmp_mae_round, '*-');
hold on
% line([1,d], [PMF_output.RMSE_baseline, PMF_output.RMSE_baseline])
plot(j, y)
hold off
legend('pmf', 'baseline')
title('MAE VS Different d')
xlabel('d')
ylabel('MAE')

% no difference...
% choose d = 3, d = 5;

%% optimize mu
d = 3;
figure
j = 1;
for mu = 0.001:0.004:0.013
    subplot(2,2,j);
    j = j + 1;
    data_matrix = cellofmatrix{2};
    [PMF_output] = PMF(data_matrix, rating_to_pred, iteration, d, weight_missing, mu, lambda, rm);
    i = 1:1:iteration;
    plot(i, PMF_output.MAE_iter_round);
    hold on
    plot(i, PMF_output.RMSE_iter_round);
    hold off
    legend('MAE', 'RMSE')
    str=['MAE & RMSE VS Iteration with Learning Rate ',num2str(mu)];
    title(str)
    xlabel('Iteration')
    ylabel('MAE & RMSE')
    
end
% choose mu = 0.005
mu = 0.005;
%% optimize lambda
figure
subplot(1,2,1);
i = 1:1:iteration;
data_matrix = cellofmatrix{5};
for lambda = -0.04:0.01:0.05
    [PMF_output] = PMF(data_matrix, rating_to_pred, iteration, d, weight_missing, mu, lambda, rm);
%     figure
%     plot(i, PMF_output.RMSE_iter);
%     hold on
    plot(i, PMF_output.MAE_iter_round);
    hold on
end
hold off
lambda = -0.04:0.01:0.05;
legend('-0.04','-0.03','-0.02','-0.01','0','0.01','0.02','0.03','0.04','0.05')
str='MAE VS Iteration with Lambda from -0.04 to 0.05';
title(str)
xlabel('Iteration')
ylabel('MAE')

subplot(1,2,2);
for lambda = -0.04:0.01:0.05
    [PMF_output] = PMF(data_matrix, rating_to_pred, iteration, d, weight_missing, mu, lambda, rm);
%     figure
%     plot(i, PMF_output.RMSE_iter);
%     hold on
    plot(i, PMF_output.RMSE_iter_round);
    hold on
end
hold off
lambda = -0.5:0.1:0.5;
legend('-0.04','-0.03','-0.02','-0.01','0','0.01','0.02','0.03','0.04','0.05')
str='RMSE VS Iteration with Lambda from -0.04 to 0.05';
title(str)
xlabel('Iteration')
ylabel('RMSE')


% don't understand how lambda works
% choose lambda = 0.01
lambda = 0.01;
%% cluster using kmeans with k=2 and k=3
d = 3;
mu = 0.005;
j = 1;
for i = 1:9:10  % 3
    tmp = 0.3 + 0.05 * i;
    data_matrix = cellofmatrix{i};
    [PMF_output] = PMF(data_matrix, rating_to_pred, iteration, d, weight_missing, mu, lambda, rm);
    tmp_label_U = kmeans(PMF_output.U, 5);
    tmp_label_V = kmeans(PMF_output.V, 5);
    
%     figure
%     subplot(1,2,1)
    subplot(2,2,j)
    j = j + 1;
    tmp1 = find(tmp_label_U(:, 1) == 1);
    tmp2 = find(tmp_label_U(:, 1) == 2);
    tmp3 = find(tmp_label_U(:, 1) == 3);
    tmp4 = find(tmp_label_U(:, 1) == 4);
    tmp5 = find(tmp_label_U(:, 1) == 5);
    
    plot3(PMF_output.U(tmp1, 1), PMF_output.U(tmp1, 2), PMF_output.U(tmp1, 3),'*')
    hold on
    plot3(PMF_output.U(tmp2, 1), PMF_output.U(tmp2, 2), PMF_output.U(tmp2, 3),'*')
    hold on
    plot3(PMF_output.U(tmp3, 1), PMF_output.U(tmp3, 2), PMF_output.U(tmp3, 3),'*')
    hold on
    plot3(PMF_output.U(tmp4, 1), PMF_output.U(tmp4, 2), PMF_output.U(tmp4, 3),'*')
    hold on
    plot3(PMF_output.U(tmp5, 1), PMF_output.U(tmp5, 2), PMF_output.U(tmp5, 3),'*')
    hold off
    grid on
    str = ['U: Item Matrix Data Distribution with Sparsity ', num2str(tmp)];
    title(str)
    
%     subplot(1,2,2)
    subplot(2,2,j)
    j = j + 1;
    tmp1 = find(tmp_label_V(:, 1) == 1);
    tmp2 = find(tmp_label_V(:, 1) == 2);
    tmp3 = find(tmp_label_V(:, 1) == 3);
    tmp4 = find(tmp_label_V(:, 1) == 4);
    tmp5 = find(tmp_label_V(:, 1) == 5);
    
    plot3(PMF_output.V(tmp1, 1), PMF_output.V(tmp1, 2), PMF_output.V(tmp1, 3),'*')
    hold on
    plot3(PMF_output.V(tmp2, 1), PMF_output.V(tmp2, 2), PMF_output.V(tmp2, 3),'*')
    hold on
    plot3(PMF_output.V(tmp3, 1), PMF_output.V(tmp3, 2), PMF_output.V(tmp3, 3),'*')
    hold on
    plot3(PMF_output.V(tmp4, 1), PMF_output.V(tmp4, 2), PMF_output.V(tmp4, 3),'*')
    hold on
    plot3(PMF_output.V(tmp5, 1), PMF_output.V(tmp5, 2), PMF_output.V(tmp5, 3),'*')
    hold off
    grid on
    str = ['V: User Matrix Data Distribution with Sparsity ', num2str(tmp)];
    title(str) 
    
end

d = 2;
figure
j = 1;
for i = 1:9:10  % 3
    tmp = 0.3 + 0.05 * i;
    data_matrix = cellofmatrix{i};
    [PMF_output] = PMF(data_matrix, rating_to_pred, iteration, d, weight_missing, mu, lambda, rm);
    tmp_label_U = kmeans(PMF_output.U, 5);
    tmp_label_V = kmeans(PMF_output.V, 5);
    
%     figure
%     subplot(1,2,1)
    subplot(2,2,j)
    j = j + 1;
    tmp1 = find(tmp_label_U(:, 1) == 1);
    tmp2 = find(tmp_label_U(:, 1) == 2);
    tmp3 = find(tmp_label_U(:, 1) == 3);
    tmp4 = find(tmp_label_U(:, 1) == 4);
    tmp5 = find(tmp_label_U(:, 1) == 5);

    plot(PMF_output.U(tmp1, 1), PMF_output.U(tmp1, 2),'*')
    hold on
    plot(PMF_output.U(tmp2, 1), PMF_output.U(tmp2, 2),'*')
    hold on
    plot(PMF_output.U(tmp3, 1), PMF_output.U(tmp3, 2),'*')
    hold on
    plot(PMF_output.U(tmp4, 1), PMF_output.U(tmp4, 2),'*')
    hold on
    plot(PMF_output.U(tmp5, 1), PMF_output.U(tmp5, 2),'*')
    hold off
    grid on
    str = ['U: Item Matrix Data Distribution with Sparsity ', num2str(tmp)];
    title(str) 
    
    subplot(2,2,j)
    j = j + 1;
    tmp1 = find(tmp_label_V(:, 1) == 1);
    tmp2 = find(tmp_label_V(:, 1) == 2);
    tmp3 = find(tmp_label_V(:, 1) == 3);
    tmp4 = find(tmp_label_V(:, 1) == 4);
    tmp5 = find(tmp_label_V(:, 1) == 5);

    plot(PMF_output.V(tmp1, 1), PMF_output.V(tmp1, 2),'*')
    hold on
    plot(PMF_output.V(tmp2, 1), PMF_output.V(tmp2, 2),'*')
    hold on
    plot(PMF_output.V(tmp3, 1), PMF_output.V(tmp3, 2),'*')
    hold on
    plot(PMF_output.V(tmp4, 1), PMF_output.V(tmp4, 2),'*')
    hold on
    plot(PMF_output.V(tmp5, 1), PMF_output.V(tmp5, 2),'*')
    hold off
    grid on
    str = ['V: User Matrix Data Distribution with Sparsity ', num2str(tmp)];
    title(str) 
end

%% using optimized parameter to get the output over 10 smalldataset
load('data.mat')

iteration = 200;
weight_missing = 0;
mu = 0.005;
lambda = 0.01;
rm = 0;
d = 3;
MAE_PMF = zeros(numel(cellofmatrix), 1);
RMSE_PMF = zeros(numel(cellofmatrix), 1);


for i = 1:1:numel(cellofmatrix)
    data_matrix = cellofmatrix{i};
    [PMF_output] = PMF(data_matrix, rating_to_pred, iteration, d, weight_missing, mu, lambda, rm);
    MAE_PMF(i ,1) = PMF_output.MAE_iter_round(iteration, 1);
    RMSE_PMF(i ,1) = PMF_output.RMSE_iter_round(iteration, 1);
end
i = 1:1:10;
figure
plot(i, MAE_PMF)
hold on
plot(i, RMSE_PMF)
hold off

% save small_output_PMF.mat MAE_PMF RMSE_PMF
```



[Back to Menu](#menu)
<br>

[Back to Menu](#menu)
<br>

[Back to Menu](#menu)
<br>

[Back to Menu](#menu)
<br>

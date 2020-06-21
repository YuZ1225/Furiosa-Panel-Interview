# Furiosa Panel Interview
This repository is used for quick access of Furiosa panel interview.
<br>

## Menu
* [Resume](https://github.com/YuZ1225/Furiosa-Panel-Interview/blob/master/resume/resume_yuzhao.pdf)
* [Different Collaborative Filtering Algorithms in Recommendation System](#cf)
* [Classification of Brazilian Names](#brz)
* [Scramble String](#scrm)
* [Edit Distance](#ed)
<br>

<a id = cf></a>
## Different Collaborative Filtering Algorithms in Recommendation System
In this project we try to use different kinds of collaborative filtering algorithms to build up a book recommendation system. We use the dataset from [kaggle](https://www.kaggle.com/philippsp/book-recommender-collaborative-filtering-shiny). In this dataset, we have 10k books and 53k users, with 99.82% sparsity. In the end we successfully build up four different kinds CF algorithms.

Here are our full report of this project: [full report](https://github.com/YuZ1225/Furiosa-Panel-Interview/blob/master/Different%20Collaborative%20Filtering%20Algorithms%20in%20Recommendation%20System/Final_report.pdf)

Here are our ppt presentation of this project: [ppt](https://github.com/YuZ1225/Furiosa-Panel-Interview/blob/master/Different%20Collaborative%20Filtering%20Algorithms%20in%20Recommendation%20System/final_presentation.pptx)

Following are some code snippets that I contributed in this project.

<details>
	<summary> Naive Bayes Collaborative Filter Code </summary>
	
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
</details>


<details>
	<summary> Probabolistic Matrix Factorization Code </summary>
	
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

%    Initialization
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

%     Gradient descent
		U = U - para_mu * (-(W .* (data_matrix - U*V')) * V + para_lambda * U);
		V = V - para_mu * (-(W .* (data_matrix - U*V'))' * U + para_lambda * V);

		tmp_result = (U*V');
		label_pred(:, i) = tmp_result(tmp_index);
		MAE_iter(i, 1) = mean(abs(label_pred(:, i) - rating_to_pred(:,3)));
		RMSE_iter(i, 1) = sqrt(immse(label_pred(:, i), rating_to_pred(:,3)));

		label_pred_round(:, i) = round(label_pred(:, i));
		MAE_iter_round(i, 1) = mean(abs(label_pred_round(:, i) - rating_to_pred(:,3)));
		RMSE_iter_round(i, 1) = sqrt(immse(label_pred_round(:, i), rating_to_pred(:,3)));
		tmp_result = [];    % clear memory
		toc;
	end

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
</details>


<details>
	<summary> Test and optimize parameter of PMF </summary>
	
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
</details>

<details>
	<summary>Some result images</summary><blockquote>
	
<details>
	<summary>Optimize latent dimension d</summary><blockquote>

![image d](https://github.com/YuZ1225/Furiosa-Panel-Interview/blob/master/Different%20Collaborative%20Filtering%20Algorithms%20in%20Recommendation%20System/optimize_d.png)
</blockquote></details>

<details>
	<summary>Optimize lambda</summary><blockquote>
	
![image lambda](https://github.com/YuZ1225/Furiosa-Panel-Interview/blob/master/Different%20Collaborative%20Filtering%20Algorithms%20in%20Recommendation%20System/optimize_lambda.png)
</blockquote></details>

<details>
	<summary>Optimize mu</summary><blockquote>
	
![image mu](https://github.com/YuZ1225/Furiosa-Panel-Interview/blob/master/Different%20Collaborative%20Filtering%20Algorithms%20in%20Recommendation%20System/optimize_mu.png)
</blockquote></details>

<details>
	<summary>PMF process</summary><blockquote>
	
![image pmf](https://github.com/YuZ1225/Furiosa-Panel-Interview/blob/master/Different%20Collaborative%20Filtering%20Algorithms%20in%20Recommendation%20System/pmf_d_3_10.png)
</blockquote></details>

<details>
	<summary>Dataset distribution in 3D</summary><blockquote>
	
![image 3d](https://github.com/YuZ1225/Furiosa-Panel-Interview/blob/master/Different%20Collaborative%20Filtering%20Algorithms%20in%20Recommendation%20System/UV_entire_in_3d.png)
</blockquote></details>

</blockquote></details>


<a id = brz></a>
## Classification of Brazilian Names
In this project we want to identify the Brazilian immigrants who are in the USA by their names. We got the dataset with 60k full names. In the end we established four different kinds of methods to identify the Brazilian names.

Here are our full report of this project: [full report](https://github.com/YuZ1225/Furiosa-Panel-Interview/blob/master/Classification%20of%20Brazilian%20Names/CS542_Final_Report.pdf)

Here are our poster presentation of this project: [poster](https://github.com/YuZ1225/Furiosa-Panel-Interview/blob/master/Classification%20of%20Brazilian%20Names/CS542_Poster_Group1_Digaai.pdf)

In this project I came up with an idea to convert the names, which are strings, into unique numbers with base 26. Then use KNN algorithms to classify the names. The result shows that using this method has a big advantage in running time, which are in seconds, while having a good accuracy rate at the same time.

Following is a code snippet that I contributed in this project.

<details>
	<summary>Name Classify through KNN Algorithm Code</summary>
	
```Matlab
%% Name Classify through KNN Algorithm

fid = fopen('dtrain - Copy.csv');
train_raw = textscan(fid,'%d%s%s%d', 'Delimiter', ',');
fclose(fid);
First_train = string(train_raw{2});
Last_train = string(train_raw{3});
label_train = train_raw{4};
train_len = numel(First_train);
data2num_train = zeros(train_len ,3);

%   Transfer name into unique number with base 26
i = 1;
while i <= train_len
	j = 2;
	je = 7; %5
	k = 2;
	ke = 7; %5
	f_len = length(First_train{i});
	l_len = length(Last_train{i});
	data2num_train(i,1) = (First_train{i}(1) - '@')*26^8;
	data2num_train(i,2) = (Last_train{i}(1) - '@')*26^8;
	while j <= f_len && je >= -1
		if First_train{i}(j) == '\'
			j = j+4;
		else
			data2num_train(i,1) = data2num_train(i,1) + (First_train{i}(j) - '`')*26^je;
			j = j+1;
			je = je-1;
		end
	end

	while k <= l_len && ke >= -1
		if Last_train{i}(k) == '\'
			k = k+4;
		else
			data2num_train(i,2) = data2num_train(i,2) + (Last_train{i}(k) - '`')*26^ke;
			k = k+1;
			ke = ke-1;
		end
	end
	i = i+1;
end
data2num_train(: ,3) = label_train(:);

fid = fopen('dtest - Copy.csv');
test_raw = textscan(fid,'%d%s%s%d', 'Delimiter', ',');
fclose(fid);
First_test = string(test_raw{2});
Last_test = string(test_raw{3});
label_test = test_raw{4};
test_len = numel(First_test);
data2num_test = zeros(test_len ,3);
i = 1;

%   Transfer name into unique number with base 26
while i <= test_len
	j = 2;
	je = 7; %5
	k = 2;
	ke = 7; %5
	f_len = length(First_test{i});
	l_len = length(Last_test{i});
	data2num_test(i,1) = (First_test{i}(1) - '@')*26^8;
	data2num_test(i,2) = (Last_test{i}(1) - '@')*26^8;
	while j <= f_len && je >= -1
		if First_test{i}(j) == '\'
			j = j+4;
		else
			data2num_test(i,1) = data2num_test(i,1) + (First_test{i}(j) - '`')*26^je;
			j = j+1;
			je = je-1;
		end
	end

	while k <= l_len && ke >= -1
		if Last_test{i}(k) == '\'
			k = k+4;
		else
			data2num_test(i,2) = data2num_test(i,2) + (Last_test{i}(k) - '`')*26^ke;
			k = k+1;
			ke = ke-1;
		end
	end
	i = i+1;
end
% data2num_test(: ,3) = label_test(:);


%% 2D 74% // 1D last name 78% // 1D first name 81%
% X = data2num_train(:,1:2);
% Y = data2num_train(:,3);
% tmp1 = find(Y == 1);
% tmp2 = find(Y == 0);
% figure;
% scatter(X(tmp1, 1)/(26^8), X(tmp1, 2)/(26^8), 8, 'o', 'r', 'filled')
% hold on
% scatter(X(tmp2, 1)/(26^8), X(tmp2, 2)/(26^8), 5, '+', 'b')
% hold off
% legend('Brazillian', 'Not Brazillian')
% title('Scatter of names in unique numbers')
% xlabel('First name in unique number')
% ylabel('Last name in unique number')
% Mdl = fitcknn(X,Y,'NumNeighbors',1,'Standardize',1);
% X2 = data2num_test(:,1:2);
% % Y2 = data2num_test(:,3);
% i = 0:1:test_len-1;
% lab = zeros(test_len+1,2);
% lab(2:end,1) = i;
% lab(2:end, 2) = predict(Mdl, X2);
% % xxx = find(lab(:) == Y2(:));
% % CCR = length(xxx)/length(X2);
% csvwrite('sample.csv', lab);

%% compare distance in last name or first name(first name first) %82.8 // increase the effective digits to 8 digits we have 83%, no increase.lack of data.
i = 0:1:test_len-1;
lab = zeros(test_len+1,2);
lab(2:end,1) = i;
X_first = data2num_train(:,1);
X_last = data2num_train(:,2);
X2_first = data2num_test(:,1);
X2_last = data2num_test(:,2);
Y = data2num_train(:,3);
[id_tmp1, d1] = knnsearch(X_first, X2_first);
[id_tmp2, d2] = knnsearch(X_last, X2_last);
lab(2:end, 2) = Y(id_tmp1);
tmp = find(d1>d2);
lab(tmp+1, 2) = Y(id_tmp2(tmp));
csvwrite('sample.csv', lab);

%% find nearer point in last name or first name(last name first) 82.2%
% i = 0:1:test_len-1;
% lab = zeros(test_len+1,2);
% lab(2:end,1) = i;
% X_first = data2num_train(:,1);
% X_last = data2num_train(:,2);
% X2_first = data2num_test(:,1);
% X2_last = data2num_test(:,2);
% Y = data2num_train(:,3);
% [id_tmp1, d1] = knnsearch(X_first, X2_first);
% [id_tmp2, d2] = knnsearch(X_last, X2_last);
% lab(2:end, 2) = Y(id_tmp2);
% tmp = find(d1<d2);
% lab(tmp+1, 2) = Y(id_tmp1(tmp));
% csvwrite('sample.csv', lab);
```
</details>



<a id = scrm></a>
## Leetcode 87. Scramble String
I noticed that if one string is a scramble of another, then they will have two properties:
* The characters in two string are the same.
* There will be a pivot point, and the two strings will only be one of the two conditions:
	<blockquote>

	|string s1 | string s2|
	|-----|-----|
	|scramble(s1) | scramble(s2)|

	or

	|string s1 | string s2|
	|-----|-----|
	|scramble(s2) | scramble(s1)|
	</blockquote>

Then we can use recurrisive method to solve this problem.

I posted my solution on the leetcode discussion board: [my solution](https://leetcode.com/problems/scramble-string/discuss/635917/easy-understand-4ms-c-recursive-solution)


<a id = ed></a>
## Leetcode 72. Edit Distance
<details>
	<summary> My solution </summary>
	
```cpp
class Solution {
public:
    int minDistance(string word1, string word2) {
	// -----BOTOM-UP DP-----
	int m = word1.size();
	int n = word2.size();
	vector<vector<int>> ans(m + 1, vector<int>(n + 1, 0));
	for(int i = 0; i < m + 1; i++){
	    ans[i][0] = i;
	}
	for(int i = 0; i < n + 1; i++){
	    ans[0][i] = i;
	}
	for(int i = 1; i < m + 1; i++){
	    for(int j = 1; j < n + 1; j++){
		if(word1[i-1] == word2[j-1]){
		    ans[i][j] = ans[i-1][j-1];
		}
		else{
		    ans[i][j] = min(min(ans[i-1][j], ans[i-1][j-1]), ans[i][j-1]) + 1;
		}
	    }
	}
	return ans[m][n];
    }
};
```
</details>

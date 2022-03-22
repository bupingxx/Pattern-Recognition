clc();
clear();

% scale [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
% set the random number seed to 0 for reproducibility
rng(0);
avg = [1 2 3 4 5 6 7 8 9 10];
scale = 0.005;

data = randn(5000, 10) + repmat(avg*scale, 5000, 1);

m = mean(data);
m1 = m / norm(m);

% do PCA, but without centering
[~, S, V] = svd(data);
S = diag(S);
e1 = V(:,1);  % 第一个特征向量，未减去平均向量

% do correct PCA with centering
newdata = data - repmat(m, 5000, 1);
[U, S, V] = svd(newdata);
S = diag(S);
new_e1 = V(:,1); % 第一个特征向量，减去平均向量

% correlation between first eigenvector(new , old) and mean
avg = avg - mean(avg);
avg = avg / norm(avg);
e1 = e1 - mean(e1);
e1 = e1 / norm(e1);
new_e1 = new_e1 - mean(new_e1);
new_e1 = new_e1 / norm(new_e1);
corr1 = avg * e1;
corr2 = e1.' * new_e1;

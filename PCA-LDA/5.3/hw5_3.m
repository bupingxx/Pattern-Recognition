clc();
clear();

% a小题 生成样本
rng(0);
x = randn(2000, 2)*[2 1;1 2];
figure(1);
scatter(x(:,1), x(:,2));    % 画图
xlim([-10,10]);
ylim([-10,10]);

% b小题 做PCA变换
c = cov(x); %协方差
m = mean(x);
data = x - repmat(m, 2000, 1); %去均值
[U, S, V] = svd(c);            %svd分解
pca_data = data * V;           
figure(2);
scatter(pca_data(:,1), pca_data(:,2));    % 画图
xlim([-10,10]);
ylim([-10,10]);

% c小题 做白化变换
pca_whiten_data = pca_data * (S)^(-0.5);
figure(3);
scatter(pca_whiten_data(:,1), pca_whiten_data(:,2));    % 画图
xlim([-5,4]);
ylim([-4,4]);
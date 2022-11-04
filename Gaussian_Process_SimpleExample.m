% 0) Statement: Gaussian Process Example  (仅适用于一维度)
% 1) 生成原函数数据 和 用于画图的50个样本点（等间隔取点）Sample Data 
clc; clear; close all;
OriginDataX = linspace(0, 10,100)';
OriginDataY = sin(OriginDataX);
SampleDataX = linspace(0, 10, 50)';
SampleDataY = sin(SampleDataX);
% 2) 生成原始数据并添加噪声 -> 观测数据    Train Data
TrainDataX = linspace(0, 10, 5)';  MuNoise = [0.0];
TrainDataY = sin(TrainDataX);      SigmaNoise = [0.01];
for i = 1 : 1 : length(TrainDataY); TrainDataY(i) = TrainDataY(i) + sqrt(SigmaNoise) * randn(1,1) + MuNoise; end
% 3) 绘制原始函数和样本点
figure(1); clf; subplot(2, 2, 1);
plot(OriginDataX, OriginDataY, 'k', 'linewidth', 2); hold on;  % 原函数
plot(TrainDataX, TrainDataY, 'k*');                  hold on;  % 取样点
xlim([0, 10]); ylim([-1.5, 1.5]); 
title('目标函数和样本点（带噪声）');  xlabel('TrainX','FontSize',12,'FontName','Times'); ylabel('Sin(TrainX)','FontSize',12,'FontName','Times');
legend('目标函数', '样本点（带噪声）', 'Location', 'southeast');
% 4) 绘制 Gaussian Process 的先验分布图
SigmaSampleX = RBF_Kernel(SampleDataX, SampleDataX);
SampleMuLine = zeros(size(SampleDataX, 1), 1);
SampleSULine = SampleMuLine + sqrt(diag(SigmaSampleX));
SampleSLLine = SampleMuLine - sqrt(diag(SigmaSampleX));
subplot(2, 2, 2);
fill([SampleDataX' fliplr(SampleDataX')], [SampleSULine' fliplr(SampleSLLine')], [200, 255, 212]/255, 'linestyle', 'none', 'FaceAlpha',0.5); hold on
plot(SampleDataX, SampleMuLine, 'color', [108, 74, 182]/255, 'linewidth', 2); hold on;
xlim([0, 10]); ylim([-1.5, 1.5]); 
title('Gaussian Process 先验'); xlabel('TrainX','FontSize',12,'FontName','Times'); ylabel('Gaussian Process Y','FontSize',12,'FontName','Times');
legend('置信区间', 'GP 均值', 'Location', 'southeast');
% 5) 根据 Train Data 求解 Sample Data的后验分布
SigmaSampleX_TrainX = RBF_Kernel(SampleDataX, TrainDataX);
SigmaTrainX_SampleX = RBF_Kernel(TrainDataX, SampleDataX);
SigmaTrainX         = RBF_Kernel(TrainDataX,  TrainDataX);
MuX = SigmaSampleX_TrainX * (SigmaTrainX \ TrainDataY);
SigmaX = SigmaSampleX - SigmaSampleX_TrainX * (SigmaTrainX \ SigmaTrainX_SampleX);
SigmaSUX = MuX + real(sqrt(diag(SigmaX)));
SigmaSLX = MuX - real(diag(SigmaX));
subplot(2, 2, 3);
fill([SampleDataX' fliplr(SampleDataX')], [SigmaSUX' fliplr(SigmaSLX')], [200, 255, 212]/255, 'linestyle', 'none', 'FaceAlpha',0.5); hold on
plot(SampleDataX, MuX, 'color', [108, 74, 182]/255, 'linewidth', 2); hold on;
plot(TrainDataX, TrainDataY, '*', 'color', [108, 74, 182]/255); hold on;
xlim([0, 10]); ylim([-1.5, 1.5]); 
title('Gaussian Process 后验'); xlabel('TrainX','FontSize',12,'FontName','Times'); ylabel('Gaussian Process Y','FontSize',12,'FontName','Times');
legend('置信区间', 'GP 均值', 'Location', 'southeast');
% 6) 进行单值的预测并绘图
TestDataX = 3.5;
SigmaTestX        = RBF_Kernel(TestDataX,  TestDataX);
SigmaTestX_TrainX = RBF_Kernel(TestDataX, TrainDataX);
SigmaTrainX_TestX = RBF_Kernel(TrainDataX, TestDataX);
MuTestX    = SigmaTestX_TrainX * (SigmaTrainX \ TrainDataY);
SigmaTestX = SigmaTestX - SigmaTestX_TrainX * (SigmaTrainX \ SigmaTrainX_TestX);
SigmaSUTestX = MuTestX + sqrt(diag(SigmaTestX));
SigmaSLTestX = MuTestX - sqrt(diag(SigmaTestX));

subplot(2, 2, 4);
fill([SampleDataX' fliplr(SampleDataX')], [SigmaSUX' fliplr(SigmaSLX')], [200, 255, 212]/255, 'linestyle', 'none', 'FaceAlpha',0.5); hold on
plot(SampleDataX, MuX, 'color', [108, 74, 182]/255, 'linewidth', 2); hold on;
xlim([0, 10]); ylim([-1.5, 1.5]);
GP_BaseX = linspace(TestDataX, TestDataX, 50);
GP_BaseY = linspace(-1.5, 1.5, 50);
plot(GP_BaseX, GP_BaseY, 'color', [141, 158, 255]/255, 'linewidth', 2); hold on;
GP_Y = 1/(sqrt(2*pi) * sqrt(SigmaTestX)) * exp(-1 * (GP_BaseY - MuTestX).^2 ./ (2 * SigmaTestX));
plot(GP_Y+TestDataX, GP_BaseY, 'color', [141, 158, 255]/255, 'linewidth', 2); hold on;
title('Gaussian Process 单点预测'); xlabel('TrainX','FontSize',12,'FontName','Times'); ylabel('Gaussian Process Y','FontSize',12,'FontName','Times');
legend('置信区间', 'GP 均值', '单点的高斯分布', 'Location', 'southeast');

function [Cov] = RBF_Kernel(X1, X2, Param1, Param2)
if nargin < 3; Param1 = 1.0^2; Param2 = 0.8; end
Cov = zeros(size(X1, 1), size(X2, 1));
for i = 1 : size(X1, 1)
    for j = 1 : size(X2, 1)
        Cov(i, j) = RBF_Covariance(X1(i), X2(j), Param1, Param2);
    end
end
end

function [Cov] = RBF_Covariance(X1, X2, Param1, Param2)
if nargin < 3; Param1 = 1.0^2; Param2 = 0.8; end
Cov = Param1^2 * exp(-1/(2*Param2^2) * (X1 - X2)^2);
end

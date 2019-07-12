clear all;close all;clc
%% 加载数据
load('../data/filmtrust.mat');
configs;
%% 数据预处理
spData=data2sparseForm(data);
[trainData,testData]=getTrainAndTestByPos(spData,7,10);
info=pretreatment( trainData,testData,trust );
[orgTrainMat,orgTestMat,cenTrainMat,cenTestMat,rowVec,colVec,avg]=dataCentralized(trainData,testData,info);
%% 数据预处理
[ U,M,Y,Z,alpha,beta,rmseOfSvdpp ] = BCDTrustSvd( cenTrainMat,cenTestMat,info,confs{2});
%% 显示结果
[rmse,predictMat]=checkRmseOfResult(orgTestMat,U,M,Y,Z,alpha,beta,rowVec,colVec,avg,info);
% disp('测试集的预测结果为：');
% predictMat
disp(sprintf('均方根误差(RMSE)为：%g',rmse));
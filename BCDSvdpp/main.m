clear all;close all;clc
%% 设置数据集编号
dataId=6;
%% 加载数据
if dataId<=5
	load(sprintf('../data/movielens%i.mat',dataId));
elseif dataId==6
    load('../data/filmtrust.mat');
end
configs;
%% 数据预处理
spData=data2sparseForm(data);
[trainData,testData]=getTrainAndTestByPos(spData,7,10);
info=pretreatment( trainData,testData );
[orgTrainMat,orgTestMat,cenTrainMat,cenTestMat,rowVec,colVec,avg]=dataCentralized(trainData,testData,info);
%% 数据预处理
[ U,M,Y,alpha,beta,rmseOfSvdpp ] = BCDSvdpp( cenTrainMat,cenTestMat,info,confs{dataId});
%% 显示结果
[rmse,predictMat]=checkRmseOfResult(orgTestMat,U,M,Y,alpha,beta,rowVec,colVec,avg,info);
% disp('测试集的预测结果为：');
% predictMat
disp(sprintf('均方根误差(RMSE)为：%g',rmse));
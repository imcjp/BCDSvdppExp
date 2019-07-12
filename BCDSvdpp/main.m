clear all;close all;clc
%% �������ݼ����
dataId=6;
%% ��������
if dataId<=5
	load(sprintf('../data/movielens%i.mat',dataId));
elseif dataId==6
    load('../data/filmtrust.mat');
end
configs;
%% ����Ԥ����
spData=data2sparseForm(data);
[trainData,testData]=getTrainAndTestByPos(spData,7,10);
info=pretreatment( trainData,testData );
[orgTrainMat,orgTestMat,cenTrainMat,cenTestMat,rowVec,colVec,avg]=dataCentralized(trainData,testData,info);
%% ����Ԥ����
[ U,M,Y,alpha,beta,rmseOfSvdpp ] = BCDSvdpp( cenTrainMat,cenTestMat,info,confs{dataId});
%% ��ʾ���
[rmse,predictMat]=checkRmseOfResult(orgTestMat,U,M,Y,alpha,beta,rowVec,colVec,avg,info);
% disp('���Լ���Ԥ����Ϊ��');
% predictMat
disp(sprintf('���������(RMSE)Ϊ��%g',rmse));
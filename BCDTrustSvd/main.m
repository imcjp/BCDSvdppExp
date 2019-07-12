clear all;close all;clc
%% ��������
load('../data/filmtrust.mat');
configs;
%% ����Ԥ����
spData=data2sparseForm(data);
[trainData,testData]=getTrainAndTestByPos(spData,7,10);
info=pretreatment( trainData,testData,trust );
[orgTrainMat,orgTestMat,cenTrainMat,cenTestMat,rowVec,colVec,avg]=dataCentralized(trainData,testData,info);
%% ����Ԥ����
[ U,M,Y,Z,alpha,beta,rmseOfSvdpp ] = BCDTrustSvd( cenTrainMat,cenTestMat,info,confs{2});
%% ��ʾ���
[rmse,predictMat]=checkRmseOfResult(orgTestMat,U,M,Y,Z,alpha,beta,rowVec,colVec,avg,info);
% disp('���Լ���Ԥ����Ϊ��');
% predictMat
disp(sprintf('���������(RMSE)Ϊ��%g',rmse));
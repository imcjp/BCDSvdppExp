function [ trainSt,testSt,rdv ] = getTrainAndTest( M,p,rdv)
%根据比例划分训练集和测试集
n=size(M,1);
if nargin<3
    rdv=[ones(floor(n*p),1);zeros(n-floor(n*p),1)];
    rdv=rdv(randperm(n),:);
end
trainSt=full(sparse(1:sum(rdv),find(rdv),1,sum(rdv),n)*M);
testSt=full(sparse(1:sum(1-rdv),find(1-rdv),1,sum(1-rdv),n)*M);
end


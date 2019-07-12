function [ trainSt,testSt ] = getTrainAndTestByPos( M,p,md)
%分组划分训练集和测试集，分为md组，第p组作为训练集
n=size(M,1);
rdv=ones(n,1);
pt=(p+1):md:n;
rdv(pt)=0;
rdv=logical(rdv);
trainSt=full(sparse(1:sum(rdv),find(rdv),1,sum(rdv),n)*M);
testSt=full(sparse(1:sum(1-rdv),find(1-rdv),1,sum(1-rdv),n)*M);
end


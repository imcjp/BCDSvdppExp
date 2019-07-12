function [ trainSt,testSt ] = getTrainAndTestByPos( M,p,md)
%���黮��ѵ�����Ͳ��Լ�����Ϊmd�飬��p����Ϊѵ����
n=size(M,1);
rdv=ones(n,1);
pt=(p+1):md:n;
rdv(pt)=0;
rdv=logical(rdv);
trainSt=full(sparse(1:sum(rdv),find(rdv),1,sum(rdv),n)*M);
testSt=full(sparse(1:sum(1-rdv),find(1-rdv),1,sum(1-rdv),n)*M);
end


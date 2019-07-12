function [ orgTrainMat,orgTestMat,cenTrainMat,cenTestMat,rowVec,colVec,avg ] = dataCentralized( trainData,testData,info )
%��ѵ�����Ͳ��Լ����Ļ�����
n=info.n;m=info.m;
V=sparse(trainData(:,1),trainData(:,2),trainData(:,3),n,m);
orgTrainMat=V;
J=info.trainData.J;
%% ��ֵ���Ļ�
avg=mean(trainData(:,3));
V=V-J*avg;
%% �����Ļ�
rowVec=sum(V,2);
t1=sum(J,2);
rowVec=rowVec./t1;
rowVec(t1==0)=0;
rowVec=full(rowVec);
V=V-spdiags(rowVec,0,n,n)*J;
%% �����Ļ�
colVec=sum(V);
t1=sum(J);
colVec=colVec./t1;
colVec(t1==0)=0;
colVec=full(colVec);
cenTrainMat=V-J*spdiags(colVec',0,m,m);
%% ���Ļ���������
V=sparse(testData(:,1),testData(:,2),testData(:,3),n,m);
orgTestMat=V;
J=info.testData.J;
V=V-J*avg;
V=V-spdiags(rowVec,0,n,n)*J;
cenTestMat=V-J*spdiags(colVec',0,m,m);
end


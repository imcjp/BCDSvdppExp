function [ rmse,predictMat ] = checkRmseOfResult(testMat,U,M,Y,Z,alpha,beta,rowVec,colVec,avg,info)
J=info.trainData.J;
trustMat=info.trustMat;
n=info.n;
m=info.m;
p=0.5;
w=(1./sum(J,2)).^p;
spdw=spdiags(w,0,n,n);
Jt=spdw*J;
omega=(1./sum(trustMat,2)).^p;
spdomega=spdiags(omega,0,n,n);
trustMat2=spdomega*trustMat;
%% º∆À„ŒÛ≤Ó÷µ
U2=U+Y*Jt'+Z*trustMat2';
[t1,t2,t3]=find(testMat);
U2s=U2(:,t1);
Ms=M(:,t2);
y1=sum(U2s.*Ms)+(alpha(t1)+beta(t2))';
y1=y1+colVec(t2)+(rowVec(t1))'+avg;
dt=t3'-y1;
err=(sum(dt.*dt));
rmse=sqrt(err/info.testData.m);
predictMat=sparse(t1,t2,y1,n,m);
end
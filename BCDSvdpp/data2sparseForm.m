function [ R,den ] = data2sparseForm( M )
%תΪϡ��洢
T=sparse(M(:,1),M(:,2),M(:,3));
t1=sum(T);
T(:,t1==0)=[];
t1=sum(T,2);
T(t1==0,:)=[];
[t1,t2,t3]=find(T);
R=[t1,t2,t3];
den=size(M,1)/size(T,1)/size(T,2);
end


function [R] = sparseProcessing(M,Y,spA,spB )
% 实现了如公式16所示的稀疏优化
% 计算sum(M*diag(M(i,:))*spA*spB*diag(Y(i,:))))
f=size(M,1);
R=((M.*repmat(M(1,:),f,1))*spA*spB.*repmat(Y(1,:),f,1));
for i=2:f
    r=((M.*repmat(M(i,:),f,1))*spA*spB.*repmat(Y(i,:),f,1));
    R=r+R;
end
end


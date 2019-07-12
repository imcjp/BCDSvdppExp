function [ pn,ind,vn ] = topKMultQ2(J,T,p,k)
% 计算Q=J'*spdiags(omega.*omega,0,n,n)*T的值
% 然后根据Z按列选取最大元素
% 输出为列主元存储格式，并且按照值从大到小排序
% 如果CTopKMult与计算机不兼容，请把代码重新编译
d1=sum(T,2);
omega=1./(d1.^p);
omega(d1==0)=0;
[pn,ind,vn ]=CTopKMult(J',T,omega.*omega,k);
end


function [ pn,ind,vn ] = topKMult(J,p,k)
% 计算Q=J'*spdiags(w.*w,0,size(J,1),size(J,1))*J的值
% 然后根据Z按列选取最大元素
% 输出为列主元存储格式，并且按照值从大到小排序
% 如果CTopKMult与计算机不兼容，请把代码重新编译
d1=sum(J,2);
w=1./(d1.^p);
w(d1==0)=0;
[pn,ind,vn ]=CTopKMult(J',J,w.*w,k);
end


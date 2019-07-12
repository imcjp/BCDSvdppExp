function [cnt,rid,val] = getSpMatCCSIds(M)
% 将稀疏矩阵按列压缩存储(CCS)格式展开
if nargout==2
[rid,t]=find(M);
elseif nargout==3
    [rid,t,val]=find(M);
end
cnt=tabulate(t);
cnt=cnt(:,2);
m=size(M,2);
if size(cnt,1)<m
    cnt=[cnt;zeros(m-size(cnt,1),1)];
end
cnt=cumsum(cnt);
cnt=[0;cnt];
end


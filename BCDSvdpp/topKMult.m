function [ pn,ind,vn ] = topKMult(J,p,k)
% ����Q=J'*spdiags(w.*w,0,size(J,1),size(J,1))*J��ֵ
% Ȼ�����Z����ѡȡ���Ԫ��
% ���Ϊ����Ԫ�洢��ʽ�����Ұ���ֵ�Ӵ�С����
% ���CTopKMult�����������ݣ���Ѵ������±���
d1=sum(J,2);
w=1./(d1.^p);
w(d1==0)=0;
[pn,ind,vn ]=CTopKMult(J',J,w.*w,k);
end


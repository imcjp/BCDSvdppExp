function [ pn,ind,vn ] = topKMultQ2(J,T,p,k)
% ����Q=J'*spdiags(omega.*omega,0,n,n)*T��ֵ
% Ȼ�����Z����ѡȡ���Ԫ��
% ���Ϊ����Ԫ�洢��ʽ�����Ұ���ֵ�Ӵ�С����
% ���CTopKMult�����������ݣ���Ѵ������±���
d1=sum(T,2);
omega=1./(d1.^p);
omega(d1==0)=0;
[pn,ind,vn ]=CTopKMult(J',T,omega.*omega,k);
end


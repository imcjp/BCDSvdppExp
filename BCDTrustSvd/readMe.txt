==��Ҫ===============================================
main.m              �㷨������
BCDTrustSvd.m          ʵ��BCDTrustSvd�㷨
==����================================================
configs.m           ��������
CTopKMult.cpp       ��A*B��Ȼ���ÿһ�б���topK��Ԫ�أ��������㡣Ȼ��ÿ�г���ϵ�������еĺ��뱣��topK֮ǰһ��
topKMult.m          ����Q=J'*spdiags(w.*w,0,size(J,1),size(J,1))*J��ֵ
topKMultQ2.m        ����Q=J'*spdiags(omega.*omega,0,n,n)*T��ֵ
data2sparseForm.m	���ݴ���Ϊϡ���ʽ
dataCentralized.m   �������Ļ�
getSpMatCCSIds.m	��ϡ�������ѹ���洢(CCS)��ʽչ��
getTrainAndTest.m,getTrainAndTestByPos.m    ��������ѵ�����Ͳ��Լ�
sparseProcessing.m  ʵ��ϡ���Ż�
pretreatment.m      ����Ԥ�����������ݼ�����Ϣ���Ա�����ʹ��
checkRmseOfResult.m     ��֤���Լ������
==���ݼ�==============================================
dataId    ���ݼ��ļ���          ���ݼ�����
1         filmtrust.mat       FilmTrust���ݼ�
======================================================
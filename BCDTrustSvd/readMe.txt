==主要===============================================
main.m              算法主函数
BCDTrustSvd.m          实现BCDTrustSvd算法
==其他================================================
configs.m           参数配置
CTopKMult.cpp       求A*B，然后对每一列保留topK个元素，其余置零。然后每列乘上系数保持列的和与保留topK之前一致
topKMult.m          计算Q=J'*spdiags(w.*w,0,size(J,1),size(J,1))*J的值
topKMultQ2.m        计算Q=J'*spdiags(omega.*omega,0,n,n)*T的值
data2sparseForm.m	数据处理为稀疏格式
dataCentralized.m   数据中心化
getSpMatCCSIds.m	将稀疏矩阵按列压缩存储(CCS)格式展开
getTrainAndTest.m,getTrainAndTestByPos.m    比例划分训练集和测试集
sparseProcessing.m  实现稀疏优化
pretreatment.m      数据预处理，处理数据集的信息，以备后续使用
checkRmseOfResult.m     验证测试集的误差
==数据集==============================================
dataId    数据集文件名          数据集描述
1         filmtrust.mat       FilmTrust数据集
======================================================
function [ Ur,Mr,Yr,alphar,betar,lastRmse] = BCDSvdpp( trainMat,testMat,info,conf)
%实现BCDSvd++算法
%% 设置参数
f=conf.f;
ku=conf.ku;
km=conf.km;
ky=conf.ky;
ka=conf.ka;
kb=conf.kb;
topK=conf.topK;
iter=conf.iter;
rd=0.5;
p=0.5;
%% 数据预处理
n=info.n;m=info.m;
J=info.trainData.J;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Ef=eye(f);
alpha=zeros(n,1);
beta=zeros(m,1);
w=(1./sum(J,2)).^p;
spdw=spdiags(w,0,n,n);
Jt=spdw*J;
transJt=Jt';
U=(randn(f,n)*rd);
M=(randn(f,m)*rd);
Y=zeros(f,m);
JI1=full(sum(J,2))+ka;
JI2=full(sum(J)')+kb;
[cnt1,rid1] = getSpMatCCSIds(J);
[cnt2,rid2] = getSpMatCCSIds(J');
[t1,t2,t3]=find(trainMat);
[cntz,ridz,valQ] = topKMult(J,p,topK);
valQ=valQ';
ridz=ridz+1;
%% 算法主体部分
times=0;
disp(['迭代次数' 9 '误差' 9 '已用时间']);
tic
[vt1,vt2]=getMse(transJt,U,M,Y,testMat,alpha,beta,ku,km,ky,ka,kb);
rmse=sqrt(vt2/info.testData.m);
lastRmse=rmse;
te=toc;
disp([num2str(times) 9 num2str(lastRmse) 9 num2str(te)]);
while times<iter
    times=times+1;
    Ur=U;
    Mr=M;
    Yr=Y;
    alphar=alpha;
    betar=beta;
    U2=Y*transJt;
    U2s=U2(:,t1);
    Ms=M(:,t2);
    y1=(sum(U2s.*Ms))';
    Vt=sparse(t1,t2,y1,n,m);
    Rt=full(M*(trainMat-Vt-spdiags(alpha,0,n,n)*J-J*spdiags(beta,0,m,m))');
    for i=1:n
        Mt=M(:,rid2(cnt2(i)+1:cnt2(i+1)));
        du=(Mt*Mt'+ku*Ef)\Rt(:,i);
        U(:,i)=du;
    end
    U2=U+U2;
    Rt=full(U2*(trainMat-spdiags(alpha,0,n,n)*J-J*spdiags(beta,0,m,m)));
    for i=1:m
        U2t=U2(:,rid1(cnt1(i)+1:cnt1(i+1)));
        dm=(U2t*U2t'+km*Ef)\Rt(:,i);
        M(:,i)=dm;
    end
    U2s=U2(:,t1);
    Ms=M(:,t2);
    y1=t3-(sum(U2s.*Ms))';
    t4=y1-beta(t2);
    Rt=full(sum(sparse(t1,t2,t4,n,m),2));
    alpha=Rt./JI1;
    t4=y1-alpha(t1);
    Rt=full(sum(sparse(t1,t2,t4,n,m)));
    beta=Rt'./JI2;
    Rt=sparseProcessing(M,Y,transJt,Jt);
    y1=y1-alpha(t1)-beta(t2);
    Rt=Rt+M*(sparse(t1,t2,y1,n,m))'*Jt;
    for i=1:m
        if cntz(i+1)>cntz(i)
            st=ridz(cntz(i)+1:cntz(i+1));
            Mt0=M(:,st);
            Mt=Mt0.*(ones(f,1)*valQ(cntz(i)+1:cntz(i+1)));
            dy=(Mt0*Mt'+ky*Ef)\Rt(:,i);
        else
            dy=Rt(:,i)/ky;
        end
        Y(:,i)=dy;
    end
    [vt1,vt2]=getMse(transJt,U,M,Y,testMat,alpha,beta,ku,km,ky,ka,kb);
    rmse=sqrt(vt2/info.testData.m);
    if rmse>=lastRmse-1e-4
        break;
    end
    lastRmse=rmse;
    te=toc;
    disp([num2str(times) 9 num2str(lastRmse) 9 num2str(te)]);
end
end
function [ E,err ] = getMse( transJt,U,M,Y,V,alpha,beta,ku,km,ky,ka,kb )
% 计算误差值
U2=U+Y*transJt;
[t1,t2,t3]=find(V);
U2s=U2(:,t1);
Ms=M(:,t2);
y1=sum(U2s.*Ms)+(alpha(t1)+beta(t2))';
dt=t3'-y1;
err=(sum(dt.*dt));
E=0.5*err+0.5*(ku*sum(sum(U.*U))+km*sum(sum(M.*M))+ky*sum(sum(Y.*Y))+ka*alpha'*alpha+kb*beta'*beta);
end



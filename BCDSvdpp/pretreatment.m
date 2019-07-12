function [ info ] = pretreatment( trainData,testData )
sz=max([max(trainData(:,1:2));max(testData(:,1:2))]);
n=sz(1);m=sz(2);
info.n=n;
info.m=m;
J=sparse(trainData(:,1),trainData(:,2),ones(size(trainData,1),1),n,m);
J=logical(J);
J2=sparse(testData(:,1),testData(:,2),ones(size(testData,1),1),n,m);
J2=logical(J2);
info.trainData.J=J;
info.trainData.m=size(trainData,1);
info.testData.J=J2;
info.testData.m=size(testData,1);
end


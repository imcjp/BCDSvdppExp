#include "mex.h"
#include <memory.h>
void heapIns(double * vn,int * pn,double x,int p,int *cn,int k){
    int i;
    int it,tg;
    double dt,dtg;
    int cn2=*cn;
    int pt;
    if (*cn == k && x>vn[0]){
        cn2--;
        vn[0]=vn[cn2];
        pn[0]=pn[cn2];
        for(i=0;i<cn2;){
            tg=0;
            dtg=vn[i];
            if(2*i+1<cn2&&vn[2*i+1]<dtg){
                tg=1;
                dtg=vn[2*i+1];
            }
            if(2*i+2<cn2&&vn[2*i+2]<dtg){
                tg=2;
            }
            if(tg){
                pt=2*i+tg;
                dt=vn[pt];vn[pt]=vn[i];vn[i]=dt;
                it=pn[pt];pn[pt]=pn[i];pn[i]=it;
                i=pt;
            }else break;
        }
    }
    if(cn2<k){
        vn[cn2]=x;
        pn[cn2]=p;
        cn2++;
        for(i=cn2-1;i;){
            pt=(i-1)/2;
            if(vn[pt]>vn[i]){
                dt=vn[pt];vn[pt]=vn[i];vn[i]=dt;
                it=pn[pt];pn[pt]=pn[i];pn[i]=it;
                i=pt;
            }else{
                break;
            }
        }
    }
    *cn=cn2;
}
void heapPop(double * vn,int * pn,int *cn){
    int i;
    int it,tg;
    int pt;
    double dt,dtg;
    int cn2=*cn-1;
    vn[0]=vn[cn2];
    pn[0]=pn[cn2];
    for(i=0;i<cn2;){
        tg=0;
        dtg=vn[i];
        if(2*i+1<cn2&&vn[2*i+1]<dtg){
            tg=1;
            dtg=vn[2*i+1];
        }
        if(2*i+2<cn2&&vn[2*i+2]<dtg){
            tg=2;
        }
        if(tg){
            pt=2*i+tg;
            dt=vn[pt];vn[pt]=vn[i];vn[i]=dt;
            it=pn[pt];pn[pt]=pn[i];pn[i]=it;
            i=pt;
        }else break;
    }
    *cn=cn2;
}

void mexFunction( int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[] )
{
    mwIndex *in1, *pn1;
    mwIndex *in2, *pn2;
    int nRow = mxGetM(prhs[0]);  
    int nCol = mxGetN(prhs[1]);
    in1 = mxGetIr(prhs[0]);
    pn1 = mxGetJc(prhs[0]);
    in2 = mxGetIr(prhs[1]);
    pn2 = mxGetJc(prhs[1]);
    double*wvn=mxGetPr(prhs[2]);
    int k=mxGetScalar(prhs[3]);
    double* tmp = (double*)mxMalloc(sizeof(double)*nRow);
    int* wn = (int*)mxMalloc(sizeof(int)*nRow);
    int* wn2 = (int*)mxMalloc(sizeof(int)*nRow);
    memset(wn,0,sizeof(int)*nRow);
    plhs[0]=mxCreateNumericMatrix(nCol+1,1,mxINT32_CLASS,mxREAL);
    int *pr1 = (int*)mxGetPr(plhs[0]);
    int *pr2 = (int*)mxMalloc(sizeof(int)*nCol*k);
    double *pr3 =(double*) mxMalloc(sizeof(double)*nCol*k);
    int rt1=0;
    int rt2;
    int rt3;
    int cn=0;
    double* hvn = (double*)mxMalloc(sizeof(double)*k);
    int* hpn = (int*)mxMalloc(sizeof(int)*k);
    double wv;
    mwIndex * pt1;
    mwIndex * pt2;
    mwIndex * pt3;
    mwIndex * pt4;
    double s1,s2;
    for(int i=0;i<nCol;++i){
        pt1=in2+pn2[i];
        pt2=in2+pn2[i+1];
        rt3=0;
        for(;pt1!=pt2;++pt1){
            wv=wvn[*pt1];
            pt3=in1+pn1[*pt1];
            pt4=in1+pn1[*(pt1)+1];
            for(;pt3!=pt4;++pt3){
                rt2=*pt3;
                if(wn[rt2]<=i){
                    wn[rt2]=i+1;
                    tmp[rt2]=wv;
                    wn2[rt3++]=rt2;
                }else tmp[rt2]+=wv;
            }
        }
        cn=0;
        s1=0;
        for (int j=0;j<rt3;++j){
            s2=tmp[wn2[j]];
            s1+=s2;
            heapIns(hvn,hpn,s2,wn2[j],&cn,k);
        }
        rt1+=cn;
        s2=0;
        for (int j=0;j<cn;++j){
            s2+=hvn[j];
        }
        s1/=s2;
        for (int j=rt1-1;cn;--j){
            pr2[j]=hpn[0];
            pr3[j]=hvn[0]*s1;
            heapPop(hvn,hpn,&cn);
        }
        pr1[i+1]=rt1;
    }
    int nz=pr1[nCol];
    plhs[1]=mxCreateNumericMatrix(nz,1,mxINT32_CLASS,mxREAL);
    plhs[2]=mxCreateDoubleMatrix(nz,1,mxREAL);
    int *pr2r = (int*)mxGetPr(plhs[1]);
    double *pr3r = mxGetPr(plhs[2]);
    memcpy(pr2r,pr2,sizeof(int)*nz);
    memcpy(pr3r,pr3,sizeof(double)*nz);
    mxFree(pr2);
    mxFree(pr3);
    mxFree(hvn);
    mxFree(hpn);
    mxFree(tmp);
    mxFree(wn);
    mxFree(wn2);
}
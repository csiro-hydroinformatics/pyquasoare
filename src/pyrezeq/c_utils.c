#include "c_utils.h"

double c_get_eps() {
    return REZEQ_EPS;
}

double c_get_nan() {
    static double zero=0.;
    double nan=zero/zero;
    return nan;
}

double c_get_inf() {
    static double zero=0.;
    double inf=1./zero;
    return inf;
}

int isnull(double x){
    return fabs(x)<REZEQ_EPS ? 1 : 0;
}

int notnull(double x){
    return 1-isnull(x);
}

int ispos(double x){
    return x>REZEQ_EPS ? 1 : 0;
}

int isneg(double x){
    return x<-REZEQ_EPS ? 1 : 0;
}

int c_find_alpha(int nalphas, double * alphas, double s0){
    int i=0;

    if(s0<=alphas[0])
        return 0;

    if(s0>alphas[nalphas-1])
        return nalphas-2;

    while(s0>alphas[i] && i<=nalphas-2){
        i++;
    }
    return i-1;
}


#include "c_utils.h"

double c_get_eps() {
    return REZEQ_EPS;
}

double c_get_nan() {
    static double zero=0.;
    double nan=zero/zero;
    return isnan(nan) ? nan : 0.0/0.0;
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


int c_get_error_message(int err_code, char message[100]){
    int ierr=0;
    int len=100;

    if(err_code == REZEQ_ERROR_INTEGRATE_WRONG_NU)
        strncpy(message, "Wrong nu", len);

    else if(err_code == REZEQ_ERROR_INTEGRATE_OUT_OF_BOUNDS)
        strncpy(message, "j index out of bounds", len);

    else if(err_code == REZEQ_ERROR_INTEGRATE_NAN_COEFF)
        strncpy(message, "NaN values in coeffs", len);

    else if(err_code == REZEQ_ERROR_INTEGRATE_NOT_CONTINUOUS)
        strncpy(message, "Approx coeffs not continuous", len);

    else if(err_code == REZEQ_ERROR_INTEGRATE_NAN_SIM)
        strncpy(message, "Simulation produces nan", len);

    else if(err_code == REZEQ_ERROR_INTEGRATE_NO_CONVERGENCE)
        strncpy(message, "Algorithm did not converge", len);

    else
        strncpy(message, "Error code not found", len);

    return ierr;

}

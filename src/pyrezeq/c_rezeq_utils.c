#include "c_rezeq_utils.h"

double c_get_eps() {
    return REZEQ_EPS;
}

double c_get_atol(){
    return REZEQ_ATOL;
}

double c_get_rtol(){
    return REZEQ_RTOL;
}

double c_get_nan() {
    /* Defines two zero variables to make sure zero/zero != 1 (gcc compile) */
    static double zero1=0.;
    static double zero2=0.;
    return zero1/zero2;
}

double c_get_inf() {
    static double zero=0.;
    double inf=1./zero;
    return inf;
}

int c_get_nfluxes_max(){
    return REZEQ_NFLUXES_MAX;
}

int notnull(double x){
    return x<0 || x>0 ? 1 : 0;
}

int isnull(double x){
    return 1-notnull(x);
}

int isequal(double x, double y, double atol, double rtol){
    return fabs(x-y)<atol+rtol*fabs(x) ? 1 : 0;
}

int notequal(double x, double y, double atol, double rtol){
    return 1-isequal(x, y, atol, rtol);
}

double sign(double x){
    return x>=0 ? 1. : -1.;
}

double sqrtabs(double x){
    return sqrt(sign(x)*x);
}


int c_find_alpha(int nalphas, double * alphas, double s0){
    int i=0;

    if(s0<alphas[0])
        return -1;

    if(s0>alphas[nalphas-1])
        return nalphas-1;

    while(s0>=alphas[i] && i<=nalphas-2){
        i++;
    }
    return i>0 ? i-1 : 0;
}


int c_get_error_message(int err_code, char message[100]){
    int ierr=0;
    int len=100;

    if(err_code == REZEQ_ERROR_INTEGRATE_WRONG_NU)
        strncpy(message, "Invalid nu", len);

    else if(err_code == REZEQ_ERROR_INTEGRATE_OUT_OF_BOUNDS)
        strncpy(message, "j index out of bounds", len);

    else if(err_code == REZEQ_ERROR_INTEGRATE_NAN_COEFF)
        strncpy(message, "NaN values in coefficients", len);

    else if(err_code == REZEQ_ERROR_INTEGRATE_NOT_CONTINUOUS)
        strncpy(message, "Approx function not continuous, please check coefficients", len);

    else if(err_code == REZEQ_ERROR_INTEGRATE_NAN_SIM)
        strncpy(message, "Simulation produces nan", len);

    else if(err_code == REZEQ_ERROR_INTEGRATE_NO_CONVERGENCE)
        strncpy(message, "Algorithm did not converge", len);

    else if(err_code == REZEQ_ERROR_NFLUXES_TOO_LARGE)
        strncpy(message, "Number of fluxes too large", len);

    else if(err_code == REZEQ_ERROR_INTEGRATE_TSTART_EQUAL_TEND)
        strncpy(message, "end time identical to start time", len);

    else if(err_code == REZEQ_QUAD_APPROX_SAMEALPHA)
        strncpy(message, "Collapsed approximation band", len);

    else if(err_code == REZEQ_QUAD_TIME_TOOLOW)
        strncpy(message, "Integration time is too low", len);

    else if(err_code == REZEQ_QUAD_FAILEDSUMCHECK)
        strncpy(message, "Coefficients sum is not consistent", len);

    else if(err_code == REZEQ_QUAD_NFLUXES_TOO_LARGE)
        strncpy(message, "Number of fluxes is too large", len);

    else if(err_code == REZEQ_QUAD_NAN_COEFF)
        strncpy(message, "Coefficient is nan", len);

    else if(err_code == REZEQ_QUAD_NOT_CONTINUOUS)
        strncpy(message, "Function is not continuous", len);

    else if(err_code == REZEQ_QUAD_NO_CONVERGENCE)
        strncpy(message, "Algorithm did not converge", len);

    else
        strncpy(message, "Error code not found", len);

    return ierr;

}

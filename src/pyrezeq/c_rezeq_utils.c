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

double c_compiler_accuracy_kahan(){
    long double s, t=3.0;
    return 1.0-(4.0/t-1.0)*t;
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

/* Code copied from
 * https://stackoverflow.com/questions/48979861/numerically-stable-method-for-solving-quadratic-equations

  diff_of_products() computes a*b-c*d with a maximum error <= 1.5 ulp

  Claude-Pierre Jeannerod, Nicolas Louvet, and Jean-Michel Muller,
  "Further Analysis of Kahan's Algorithm for the Accurate Computation
  of 2x2 Determinants". Mathematics of Computation, Vol. 82, No. 284,
  Oct. 2013, pp. 2245-2264
 */
double diff_of_products(double a, double b, double c, double d)
{
    double w = d*c;
    double e = fma(-d, c, w);
    double f = fma(a, b, -w);
    return f+e;
}

int c_discrimin(double a, double b, double c, double discr[2]){
    double Delta = isnull(b*b-4.*a*c) ? 0. : diff_of_products(b, b, 4.*a, c);
    double qD = sqrt(fabs(Delta))/2;
    discr[0] = Delta;
    discr[1] = qD;
    return qD>=0 ? 0 : REZEQ_UTILS_QD_NEGATIVE;
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

    else if(err_code == REZEQ_UTILS_QD_NEGATIVE)
        strncpy(message, "qD value is negative", len);

    else
        strncpy(message, "Error code not found", len);

    return ierr;

}

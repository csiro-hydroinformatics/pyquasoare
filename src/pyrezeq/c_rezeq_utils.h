#ifndef __REZEQ_UTILS__
#define __REZEQ_UTILS__

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

/* Define Error messages */
#define REZEQ_ERROR 100000

#define REZEQ_ERROR_INTEGRATE_WRONG_NU 1
#define REZEQ_ERROR_INTEGRATE_OUT_OF_BOUNDS 2
#define REZEQ_ERROR_INTEGRATE_NAN_COEFF 3
#define REZEQ_ERROR_INTEGRATE_NOT_CONTINUOUS 4
#define REZEQ_ERROR_INTEGRATE_NAN_SIM 5
#define REZEQ_ERROR_INTEGRATE_NO_CONVERGENCE 6
#define REZEQ_ERROR_INTEGRATE_TSTART_EQUAL_TEND 7

#define REZEQ_ERROR_NFLUXES_TOO_LARGE 10


#define REZEQ_QUAD_APPROX_SAMEALPHA 101
#define REZEQ_QUAD_TIME_TOOLOW 102
#define REZEQ_QUAD_FAILEDSUMCHECK 103
#define REZEQ_QUAD_NFLUXES_TOO_LARGE 104
#define REZEQ_QUAD_INTEGRATE_NAN_COEFF 105
#define REZEQ_QUAD_NOT_CONTINUOUS 106
#define REZEQ_QUAD_NO_CONVERGENCE 107
#define REZEQ_QUAD_NAN_COEFF 108

#define REZEQ_UTILS_QD_NEGATIVE 501

/* Define small number */
#define REZEQ_EPS 1e-15
#define REZEQ_ATOL 1e-7
#define REZEQ_RTOL 1e-5

#define REZEQ_PI  3.1415926535897932384626433832795028841971693993751

/* Define maximum number of fluxes */
#define REZEQ_NFLUXES_MAX 20

double c_get_eps();
double c_get_atol();
double c_get_rtol();
double c_get_inf();
double c_get_nan();
int c_get_nfluxes_max();
double c_compiler_accuracy_kahan();

int isnull(double x);
int notnull(double x);
int isequal(double x, double y, double atol, double rtol);
int notequal(double x, double y, double atol, double rtol);

double diff_of_products(double a, double b, double c, double d);
double sqrtabs(double x);

int c_discrimin(double a, double b, double c, double discr[2]);

int c_find_alpha(int nalphas, double * alphas, double s0);

int c_get_error_message(int err_code, char message[100]);

#endif

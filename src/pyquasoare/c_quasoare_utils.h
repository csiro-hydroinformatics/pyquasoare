#ifndef __QUASOARE_UTILS__
#define __QUASOARE_UTILS__

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <errno.h>

/* Define Error messages */
#define QUASOARE_ERROR 100000

#define QUASOARE_APPROX_SAMEALPHA 101
#define QUASOARE_TIME_TOOLOW 102
#define QUASOARE_FAILEDSUMCHECK 103
#define QUASOARE_NFLUXES_TOO_LARGE 104
#define QUASOARE_INTEGRATE_NAN_COEFF 105
#define QUASOARE_NOT_CONTINUOUS 106
#define QUASOARE_NO_CONVERGENCE 107
#define QUASOARE_NAN_COEFF 108
#define QUASOARE_NONINCREASING_NODES 109
#define QUASOARE_NAN_SCALING 109

#define QUASOARE_UTILS_QD_NEGATIVE 501

#define QUASOARE_BENCH_NSUBDIV_TOO_HIGH 701
#define QUASOARE_BENCH_PARAMS_OUT_OF_BOUNBDS 702
#define QUASOARE_BENCH_INITIALISATION_OUT_OF_BOUNBDS 702

/* Define small number */
#define QUASOARE_EPS 1e-12
#define QUASOARE_ATOL 1e-7
#define QUASOARE_RTOL 1e-5

/* Precise value of pi */
#define QUASOARE_PI  3.1415926535897932384626433832795028841971693993751

/* Define maximum number of fluxes */
#define QUASOARE_NFLUXES_MAX 20

double c_get_inf();
double c_get_nan();
int c_get_nfluxes_max();
double c_compiler_accuracy_kahan();

int isnull(double x);
int notnull(double x);
int isequal(double x, double y, double atol, double rtol);
int notequal(double x, double y, double atol, double rtol);

double diff_of_products(double a, double b, double c, double d);

int c_quad_constants(double a, double b, double c, double values[3]);
double c_eta_fun(double x, double Delta);
double c_omega_fun(double x, double Delta);
int c_find_alpha(int nalphas, double * alphas, double s0);

int c_get_error_message(int err_code, char message[100]);

#endif

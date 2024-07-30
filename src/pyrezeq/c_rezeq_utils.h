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

#define REZEQ_ERROR_NFLUXES_TOO_LARGE 10

/* Define small number */
#define REZEQ_EPS 1e-9

#define REZEQ_PI 3.1415926535897936

/* Define maximum number of fluxes */
#define REZEQ_NFLUXES_MAX 20

double c_get_eps();
double c_get_inf();
double c_get_nan();
int c_get_nfluxes_max();

int isnull(double x);
int notnull(double x);
int ispos(double x);
int isneg(double x);
int isequal(double x, double y);
int notequal(double x, double y);

int c_find_alpha(int nalphas, double * alphas, double s0);

int c_get_error_message(int err_code, char message[100]);

#endif

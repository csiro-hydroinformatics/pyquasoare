#ifndef __REZEQ_UTILS__
#define __REZEQ_UTILS__

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

/* Define Error message */
#define REZEQ_ERROR 100000

/* Define small number */
#define REZEQ_EPS 1e-10

#define REZEQ_PI 3.1415926535897936

double c_get_eps();
double c_get_inf();
double c_get_nan();

int isnull(double x);
int notnull(double x);
int ispos(double x);
int isneg(double x);

int c_find_alpha(int nalphas, double * alphas, double s0);

#endif

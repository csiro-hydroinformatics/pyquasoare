#ifndef __REZEQ__
#define __REZEQ__

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

/* Define Error message */
#define REZEQ_ERROR 123000

/* Define small number */
#define REZEQ_EPS 1e-10

double c_integrate_forward(double t0, double u0, double a, double b, double t);

double c_integrate_inverse(double t0, double u0, double a, double b, double u);

int c_find_alpha(int nalphas, double * alphas, double u0);

int c_increment_fluxes(int nalphas, int nfluxes,
                        double * scalings,
                        double * a_matrix_noscaling,
                        double * b_matrix_noscaling,
                        int jalpha,
                        double aoj,
                        double boj,
                        double t0,
                        double t1,
                        double u0,
                        double u1,
                        double * fluxes);

int c_integrate(int nalphas, int nfluxes, double delta,
                            double * alphas,
                            double * scalings,
                            double * a_matrix_noscaling,
                            double * b_matrix_noscaling,
                            double u0,
                            double * u1,
                            double * fluxes);

int c_run(int nalphas, int nfluxes, int nval, double delta,
                            double * alphas,
                            double * scalings,
                            double * a_matrix_noscaling,
                            double * b_matrix_noscaling,
                            double u0,
                            double * u1,
                            double * fluxes);


#endif

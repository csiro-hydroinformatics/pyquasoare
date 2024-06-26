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

double c_get_eps();

double c_approx_fun(double nu, double a, double b, double c, double s);

double c_integrate_forward(double nu, double a, double b, double c,
                        double t0, double s0,
                        double t);

double c_integrate_inverse(double nu, double a, double b, double c,
                                double s0, double s1);

int c_find_alpha(int nalphas, double * alphas, double s0);

int c_increment_fluxes(int nfluxes,
                        double * scalings,
                        double nu,
                        double * a_vector_noscaling,
                        double * b_vector_noscaling,
                        double * c_vector_noscaling,
                        double aoj,
                        double boj,
                        double coj,
                        double t0,
                        double t1,
                        double s0,
                        double s1,
                        double * fluxes);

int c_integrate(int nalphas, int nfluxes, double delta,
                            double * alphas,
                            double * scalings,
                            double * nu_vector,
                            double * a_matrix_noscaling,
                            double * b_matrix_noscaling,
                            double * c_matrix_noscaling,
                            double s0,
                            double * s1,
                            double * fluxes);

int c_run(int nalphas, int nfluxes, int nval, double delta,
                            double * alphas,
                            double * scalings,
                            double * nu_vector,
                            double * a_matrix_noscaling,
                            double * b_matrix_noscaling,
                            double * c_matrix_noscaling,
                            double s0,
                            double * s1,
                            double * fluxes);

#endif

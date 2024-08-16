#ifndef __REZEQ_INTEG__
#define __REZEQ_INTEG__

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

#include "c_rezeq_utils.h"

double c_quad_fun(double a, double b, double c, double s);
double c_quad_grad(double a, double b, double c, double s);

int c_quad_steady(double a, double b, double c, double steady[2]);

int c_quad_coefficients(int islin, double a0, double a1,
                            double f0, double f1, double fm,
                            double coefs[3]);

double c_quad_delta_t_max(double a, double b, double c,
                            double Delta, double qD, double s0);

double c_quad_forward(double a, double b, double c,
                            double Delta, double qD,
                            double t0, double s0, double t);

double c_quad_inverse(double a, double b, double c,
                            double Delta, double qD,
                            double s0, double s1);

int c_quad_fluxes(int nfluxes,
                        double * aj_vector,
                        double * bj_vector,
                        double * cj_vector,
                        double aoj, double boj, double coj,
                        double t0, double t1, double s0, double s1,
                        double * fluxes);

int c_quad_integrate(int nalphas, int nfluxes,
                            double * alphas, double * scalings,
                            double * a_matrix_noscaling,
                            double * b_matrix_noscaling,
                            double * c_matrix_noscaling,
                            double t0,
                            double s0,
                            double delta,
                            int *niter, double * s1, double * fluxes);

int c_quad_model(int nalphas, int nfluxes, int nval, double delta,
                            double * alphas, double * scalings,
                            double * a_matrix_noscaling,
                            double * b_matrix_noscaling,
                            double * c_matrix_noscaling,
                            double s0, int * niter,
                            double * s1, double * fluxes);

#endif

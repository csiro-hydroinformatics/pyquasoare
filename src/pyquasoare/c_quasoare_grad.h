#ifndef __QUASOARE_GRAD__
#define __QUASOARE_GRAD__

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

#include "c_quasoare_utils.h"
#include "c_quasoare_core.h"

#define QUASOARE_GRAD_NPIECEWISEPARAMS 2
#define QUASOARE_GRAD_NFLUXES_POTENTIAL 2
#define QUASOARE_GRAD_NFLUXES_INPUTS 3
#define QUASOARE_GRAD_NFLUXES_OUTPUTS 6
#define QUASOARE_GRAD_NDERIV QUASOARE_GRAD_NPIECEWISEPARAMS*(QUASOARE_GRAD_NFLUXES_INPUTS+1)

#define QUASOARE_GRAD_IFLUX_INFLOW 0
#define QUASOARE_GRAD_IFLUX_DEMAND 1
#define QUASOARE_GRAD_IFLUX_OUTFLOW 2
#define QUASOARE_GRAD_IFLUX_BYPASS 3
#define QUASOARE_GRAD_IFLUX_SHORTFALL 4
#define QUASOARE_GRAD_IFLUX_TOTALOUT 5

#define QUASOARE_GRAD_VARTYPE_CAPACITY 0
#define QUASOARE_GRAD_VARTYPE_SCALING_INFLOW 1
#define QUASOARE_GRAD_VARTYPE_SCALING_DEMAND 2
#define QUASOARE_GRAD_VARTYPE_SCALING_OUTFLOW 3

double c_grad_forward(double M, double S, double t0, double x0, double t);

double c_grad_inverse(double M, double S, double x0, double x1);


int c_grad_fluxes_increment(double timestep,
                           double gamma,
                           double * Qstars,
                           double * mj_vector,
                           double * sj_vector,
                           double M, double S,
                           double t0, double t1,
                           double x0, double x1,
                           double *fluxes);

int c_grad_integrate(int nalphas, int nreservoirs, int nvariables,
                     double * alphas,
                     double timestep,
                     double gamma,
                     double *input_mapping,
                     double * Qstars,
                     double * flux_scalings,
                     double * coefs_linear,
                     double t0,
                     double x0,
                     double *dx0,
                     int current_reservoir_id,
                     int *variables_reservoir_id,
                     int *variables_type_id,
                     int *niter,
                     double * x1,
                     double * dx1,
                     double * fluxes,
                     double * dfluxes);

int c_grad_reservoir(int nalphas, int nreservoirs, int nvariables, int nval,
                    int errors,
                    double * alphas,
                    double timestep,
                    double gamma,
                    double * input_mapping,
                    double * flux_scalings,
                    double x0,
                    double * dx0,
                    int current_reservoir_id,
                    int *variables_reservoir_id,
                    int *variables_type_id,
                    double * Qstars,
                    double * coefs_linear,
                    int * niter,
                    double * x1,
                    double * dx1,
                    double * fluxes,
                    double * dfluxes);

int c_grad_model(int nalphas, int nreservoirs, int nvariables, int nval,
                 int errors,
                 double * alphas,
                 double timestep,
                 double * gammas,
                 double * input_mapping,
                 double * flux_scalings,
                 double x0,
                 double * dx0,
                 int *variables_reservoir_id,
                 int *variables_type_id,
                 double * Qstars,
                 double * coefs_linear,
                 double * x1,
                 double * dx1,
                 double * fluxes,
                 double * dfluxes);

#endif

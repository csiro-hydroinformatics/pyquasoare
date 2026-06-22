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
#define QUASAORE_GRAD_NDERIV  QUASOARE_GRAD_PIECEWISE_PWARAMS * (QUASAORE_GRAD_NFLUXES_INPUTS + 1)

double c_grad_forward(double M, double S, double t0, double x0, double t);

double c_grad_inverse(double M, double S, double x0, double x1);


int c_grad_fluxes_increment(double timestep,
                           double gamma,
                           double Qstars[QUASOARE_GRAD_NFLUXES_POTENTIAL],
                           double mj_vector[QUASOARE_GRAD_NFLUXES_INPUTS],
                           double sj_vector[QUASOARE_GRAD_NFLUXES_INPUT3],
                           double M, double S,
                           double t0, double t1,
                           double x0, double x1,
                           double fluxes[QUASAORE_GRAD_NFLUXES_OUTPUTS]);

int c_grad_integrate(int nalphas, int nreservoirs, int nvariables,
                     double * alphas,
                     double timestep,
                     double gamma,
                     double *input_mapping,
                     double Qstars[QUASOARE_GRAD_NFLUXES_POTENTIAL],
                     double flux_scalings[QUASOARE_GRAD_NFLUXES_INPUTS],
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
                     double fluxes[QUASOARE_NFLUXES_OUTPUTS],
                     double * dfluxes);

int c_grad_reservoir(int nalphas, int nreservoirs, int nvariables, int nval,
                    int errors,
                    double * alphas,
                    double timestep,
                    double gamma,
                    double * input_mapping,
                    double flux_scalings[QUASOARE_GRAD_NFLUXES_INPUTS],
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

int c_grad_model(int nalphas, nreservoirs, int nvariables, int nval,
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

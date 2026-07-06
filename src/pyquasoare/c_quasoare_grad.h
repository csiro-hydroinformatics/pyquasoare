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

#define QUASOARE_GRAD_INPUT_MAPPING_LASTDIM 3
#define QUASOARE_GRAD_INPUT_MAPPING_MISSINGRESERVOIRID 9999

#define QUASOARE_GRAD_IFLUX_INFLOW 0
#define QUASOARE_GRAD_IFLUX_DEMAND 1
#define QUASOARE_GRAD_IFLUX_DRAINAGE 2
#define QUASOARE_GRAD_IFLUX_BYPASS 3
#define QUASOARE_GRAD_IFLUX_SHORTFALL 4
#define QUASOARE_GRAD_IFLUX_OUTFLOW 5

#define QUASOARE_GRAD_VARTYPE_NUMBER 4
#define QUASOARE_GRAD_VARTYPE_CAPACITY 0
#define QUASOARE_GRAD_VARTYPE_SCALING_INFLOW 1
#define QUASOARE_GRAD_VARTYPE_SCALING_DEMAND 2
#define QUASOARE_GRAD_VARTYPE_SCALING_DRAINAGE 3

double c_grad_forward(double M, double S, double t0, double x0, double t);

double c_grad_inverse(double M, double S, double x0, double x1);

int c_grad_integrate(int nalphas, unsigned int nreservoirs, unsigned int nvariables,
                     double *alphas,
                     double timestep,
                     double gamma,
                     unsigned int *input_mapping,
                     double *input_mapping_weights,
                     double * Qstars,
                     double * dQstars_ext,
                     double * flux_scalings,
                     double * coefs_linear,
                     unsigned int current_reservoir_id,
                     unsigned int *variables_reservoir_id,
                     unsigned int *variables_type_id,
                     unsigned int *niter,
                     double * x,
                     double * dx,
                     double * fluxes,
                     double * dfluxes);

int c_grad_reservoir(int nalphas, unsigned int nval,
                    unsigned int errors,
                    double * alphas,
                    double timestep,
                    double gamma,
                    double * flux_scalings,
                    double x0,
                    double * Qstars,
                    double * dQstars_ext,
                    double * coefs_linear,
                    unsigned int * niter,
                    double * x,
                    double * dx,
                    double * fluxes,
                    double * dfluxes);

int c_grad_model(int nalphas, unsigned int nreservoirs, unsigned int nvariables, unsigned int nval,
                 unsigned int errors,
                 double * alphas,
                 double timestep,
                 double * gammas,
                 unsigned int * input_mapping,
                 double *input_mapping_weights,
                 double * flux_scalings,
                 double *x0,
                 unsigned int *variables_reservoir_id,
                 unsigned int *variables_type_id,
                 double * Qstars,
                 double * dQstars_ext,
                 double * coefs_linear,
                 double * x,
                 double * dx,
                 double * fluxes,
                 double * dfluxes);

#endif

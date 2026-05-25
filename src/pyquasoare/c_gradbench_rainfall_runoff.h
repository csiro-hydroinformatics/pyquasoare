#ifndef __GRADBENCH_RAINFALL_RUNOFF__
#define __GRADBENCH_RAINFALL_RUNOFF__

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

#include "c_gradbench_quadrouting.h"

int rr_prod_runtimestep(double qstar1, double qstar2,
                       double timestep, double gamma,
                       double x0, double * state, double * flux);

int rr_route_runtimestep(double qstar1, double qstar2,
                        double timestep, double gamma,
                        double x0, double * state, double * flux);

int c_gradbench_rainfall_runoff(int nval,
                                double timestep,
                                double x0_rr_prod,
                                double x0_rr_route,
                                double gamma_rr_prod,
                                double gamma_rr_route,
                                double demand_rr_route,
                                double * pin,
                                double * ein,
                                double * states_rr_prod,
                                double * states_rr_route,
                                double * qout_rr_prod,
                                double * qout_rr_route);

#endif

#ifndef __GRADBENCH_QUADROUTING__
#define __GRADBENCH_QUADROUTING__

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

int gb_notnull(double x);

int gb_isnull(double x);

double gb_omega_fun(double x, double Delta);

int gb_constants(double a, double b, double c, double constants[4]);

double gb_forward(double a, double b, double c, double constants[4],
                  double timestep, double s0);

double gb_integrate_x(double a, double b, double c, double constants[4],
                      double timestep, double s0, double s1,
                      double integ[2]);

int ro_route_runtimestep(double qstar1, double timestep, double gamma,
                         double x0, double * state, double * flux);

int c_gradbench_quadrouting(int nval, int nreaches,
                            double timestep,
                            double * x0_ro_route,
                            double * gammas_ro_route,
                            double * qin,
                            double * states_ro_route,
                            double * qout_ro_route);

#endif

#ifndef __GRADBENCH_QUADROUTING__
#define __GRADBENCH_QUADROUTING__

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

int c_gradbench_quadrouting(int nval, int nreaches,
                            double timestep,
                            double * x0_ro_route,
                            double * gammas_ro_route,
                            double * qin,
                            double * states_ro_route,
                            double * qout_ro_route);

#endif

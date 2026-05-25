#ifndef __GRADBENCH_SEMIDIST__
#define __GRADBENCH_SEMIDIST__

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

#include "c_gradbench_quadrouting.h"
#include "c_gradbench_rainfall_runoff.h"
#include "c_gradbench_semidist.h"

int c_gradbench_semidist(int nval, int nreaches,
                         double timestep,
                         double x0_rr_prod,
                         double x0_rr_route,
                         double x0_ro_route,
                         double * areas,
                         double * gammas_rr_prod,
                         double * gammas_rr_route,
                         double * gammas_ro_route,
                         double * demands_rr_route,
                         double * pin,
                         double * ein,
                         double * states_rr_prod,
                         double * states_rr_route,
                         double * states_ro_route,
                         double * qout_rr_prod,
                         double * qout_rr_route,
                         double * qout_ro_route);

#endif

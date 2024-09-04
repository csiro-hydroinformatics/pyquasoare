#ifndef __NONLINROUT__
#define __NONLINROUT__

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

#include "c_rezeq_utils.h"

int c_quadrouting(int nval, double timestep,
                        double theta, double q0,
                        double s0, double *inflows, double * outflows);

int c_nonlinrouting(int nval, int nsubdiv, double timestep,
                        double theta, double nu, double q0,
                        double s0, double *inflows, double * outflows);

#endif

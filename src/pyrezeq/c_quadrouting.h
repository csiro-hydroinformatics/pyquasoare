#ifndef __QUADROUT__
#define __QUADROUT__

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

/* Define Error message */
#define QUADROUT_ERROR 456000

int c_quadrouting(int nval, double delta, double theta, double q0,
                        double s0, double *inflow, double * outflow);

#endif

#ifndef __NONLINROUT__
#define __NONLINROUT__

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

/* Define Error message */
#define NONLINROUT_ERROR 456000

int c_nonlinrouting(int nval, int nsubdiv, double delta,
                        double theta, double nu, double q0,
                        double s0, double *inflows, double * outflows);

#endif

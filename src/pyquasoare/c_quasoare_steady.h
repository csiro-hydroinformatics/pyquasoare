#ifndef __QUASOARE_STEADY__
#define __QUASOARE_STEADY__

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

#include "c_quasoare_utils.h"

int c_quad_steady(double a, double b, double c, double steady[2]);

int c_quad_steady_scalings(int nalphas, int nfluxes, int nscalings,
                           double * alphas,
                           double * coefs,
                           double * scalings,
                           double * buff,
                           double * steady);

#endif

#ifndef __REZEQ_STEADY__
#define __REZEQ_STEADY__

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

#include "c_integ.h"

int c_steady_state(double nu, double a, double b, double c, double steady[2]);

#endif

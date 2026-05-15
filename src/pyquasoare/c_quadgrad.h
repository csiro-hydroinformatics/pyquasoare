#ifndef __QUADGRAD__
#define __QUADGRAD__

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

int c_quadgrad_run(int nval, int nreaches,
                   double timestep,
                   double x0,
                   double * gammas,
                   double * qin,
                   double * states,
                   double * qout);

#endif

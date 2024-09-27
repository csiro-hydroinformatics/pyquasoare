#ifndef __GR4JPROD__
#define __GR4JPROD__

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

#include "c_quasoare_utils.h"

int c_gr4jprod(int nval, int nsubdiv, double X1,
                        double s0,
                        double *inputs,
                        double * outputs);

#endif

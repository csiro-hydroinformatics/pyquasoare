#ifndef __GR4JPROD__
#define __GR4JPROD__

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

/* Define Error message */
#define GR4JPROD_ERROR 457000

int c_gr4jprod(int nval, int nsubdiv,double delta, double X1,
                        double s0,
                        double *rain, double *pet,
                        double * storage, double *actual_et, double * effective_rain);
#endif

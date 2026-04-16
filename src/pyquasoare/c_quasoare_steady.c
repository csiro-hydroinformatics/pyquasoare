#include "c_quasoare_steady.h"

int c_quad_steady(double a, double b, double c, double steady[2]){
    double q, x1, x2;
    double signb = b<0 ? -1. : 1.;
    double constants[3], Delta;

    c_quad_constants(a, b, c, constants);
    Delta = constants[0];

    steady[0] = c_get_nan();
    steady[1] = c_get_nan();

    if(notnull(a)){
        if(isnull(Delta)){
            steady[0] = -b/2./a;
        }
        else if(Delta>=0){
            q = -0.5*(b+signb*sqrt(Delta));
            x1 = q/a;
            x2 = c/q;
            steady[0] = x1 < x2 ? x1 : x2;
            steady[1] = x1 < x2 ? x2 : x1;
        }
    }
    else {
        if(notnull(b)){
            steady[0] = -c/b;
        }
    }
    return 0;
}

int c_quad_steady_scalings(int nalphas, int nfluxes, int nscalings,
                           double * alphas,
                           double * coefs,
                           double * scalings,
                           double * buff,
                           double * steady) {
    int i, j, jj, k, idx;
    double al0, al1, sc;
    double a, b, c;
    double s, stdy[2];

    int ierr = 0;
    int ncols = 2 * nalphas + 2;

    /* Here are the dimensions of the input arrays:
     * alphas : [nalphas]
     * coefs : [nfluxes x nalphas-1 x 3]
     * scalings : [nscalings x nfluxes]
     * buff : [2 * nalphas + 1]
     */

    for(i = 0; i < nscalings; i ++) {

        for(j = -1; j < nalphas; j++) {
            /* Compute quasoare coefficients */
            a = 0;
            b = 0;
            c = 0;

            jj = j < 0 ? 0 : j > nalphas - 2 ? nalphas - 2 : j;

            for(k = 0; k < nfluxes; k++) {
                sc = scalings[i * nfluxes + k];
                idx = (nalphas - 1) * 3 * k + 3 * jj;
                a += sc * coefs[idx];
                b += sc * coefs[idx + 1];
                c += sc * coefs[idx + 2];
            }

            /* Get bounds and adjust coefficients for extrapolation */
            if(i == -1) {
                al0 = -c_get_inf();
                al1 = alphas[j];
                /* linear function going through (al1, f(al1)) */
                c = a * al1 * al1 + b * al1 + c - 2 * a * al1;
                b = 2 * a;
                a = 0;
            }
            else if (i == nalphas - 1) {
                al0 = alphas[j];
                al1 = c_get_inf();
                /* linear function going through (al0, f(al0)) */
                c = a * al0 * al0 + b * al0 + c - 2 * a * al0;
                b = 2 * a;
                a = 0;
            }
            else {
                al0 = alphas[j];
                al1 = alphas[j + 1];
            }

            /* computes solutions */
            c_quad_steady(a, b, c, stdy);

            /* Check solution belongs to interval */
            for(k = 0; k < 2; k ++) {
                s = stdy[k];
                idx = 2 * (j + 1) + k;
                buff[idx] = s >= al0 && s < al1 ? s : c_get_nan();
            }
        }

        /* sort values */
        qsort(buff, sizeof(buff) / sizeof(*buff), sizeof(*buff),
              c_double_compare);

        /* store */
        fprintf(stdout, "\n");
        for(j = 0; j < ncols; j ++) {
            steady[i * ncols + j] = buff[j];
            fprintf(stdout, "%0.2f ", buff[j]);
        }
    }

    return ierr;
}



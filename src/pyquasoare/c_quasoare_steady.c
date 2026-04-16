#include "c_quasoare_steady.h"

int c_quad_steady(double a, double b, double c, double steady[2]){
    double q, x1, x2;
    double signb = b < 0 ? -1. : 1.;
    double constants[3], Delta;

    c_quad_constants(a, b, c, constants);
    Delta = constants[0];

    steady[0] = c_get_nan();
    steady[1] = c_get_nan();

    if(notnull(a)){
        if(isnull(Delta)) {
            steady[0] = constants[2];
        }
        else if(Delta > 0) {
            q = -0.5 * (b + signb * sqrt(Delta));
            x1 = q/a;
            x2 = c/q;
            steady[0] = x1 < x2 ? x1 : x2;
            steady[1] = x1 < x2 ? x2 : x1;
        }
    }
    else {
        if(notnull(b)){
            steady[0] = -c / b;
        }
    }
    return 0;
}


/* Dimensions of the input arrays:
 * alphas : [nalphas]
 * coefs : [nfluxes x nalphas-1 x 3]
 * flux_scalings : [nflux_scalings x nfluxes]
 * buff : [2 * nalphas + 1]
 */
int c_quad_steady_flux_scalings(int nalphas, int nfluxes, int nflux_scalings,
                           double * alphas,
                           double * coefs,
                           double * flux_scalings,
                           double * buff,
                           double * steady) {
    int i, j, jj, k, idx;
    double al0, al1, sc;
    double a, b, c;
    double s, stdy[2];

    int ierr = 0;
    int ncols = 2 * nalphas + 2;
    int sz = sizeof(buff[0]);

    for(i = 0; i < nflux_scalings; i ++) {

        fprintf(stdout, "\n");
        for(j = -1; j < nalphas; j++) {
            /* Compute quasoare coefficients */
            a = 0;
            b = 0;
            c = 0;

            jj = j < 0 ? 0 : j > nalphas - 2 ? nalphas - 2 : j;

            for(k = 0; k < nfluxes; k++) {
                sc = flux_scalings[i * nfluxes + k];
                idx = (nalphas - 1) * 3 * k + 3 * jj;
                a += sc * coefs[idx];
                b += sc * coefs[idx + 1];
                c += sc * coefs[idx + 2];
            }

            /* Get bounds and adjust coefficients for extrapolation */
            if(j == -1) {
                al0 = -c_get_inf();
                al1 = alphas[0];
                /* linear function going through (al1, f(al1)) */
                c = (a  * al1 + b - 2 * a) * al1 + c;
                b = 2 * a;
                a = 0;
            }
            else if (j == nalphas - 1) {
                al0 = alphas[j];
                al1 = c_get_inf();
                /* linear function going through (al0, f(al0)) */
                c = (a * al0  + b - 2 * a) * al0 + c;
                b = 2 * a;
                a = 0;
            }
            else {
                al0 = alphas[j];
                al1 = alphas[j + 1];
            }

            /* computes solutions */
            c_quad_steady(a, b, c, stdy);
            fprintf(stdout, "\t%2d [%0.2f, %0.2f]: %0.2f\n", j, al0, al1, stdy[0]);

            /* Check that solutions belongs to interval
             * otherwise returns the max double to ensure
             * that it remains above other values when sorting */
            for(k = 0; k < 2; k ++) {
                s = stdy[k];
                idx = 2 * (j + 1) + k;
                buff[idx] = s >= al0 && s < al1 ? s : DBL_MAX;
            }
        }

        /* sort values */
        qsort(buff, ncols, sz, c_double_compare);

        /* store */
        for(j = 0; j < ncols; j ++) {
            s = buff[j];
            s = s < DBL_MAX - 1 ? s : c_get_nan();
            steady[i * ncols + j] = s;
        }
    }

    return ierr;
}



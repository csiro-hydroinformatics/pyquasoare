
#include <stdio.h>
#include "../../src/pyquasoare/c_quasoare_utils.h"
#include "../../src/pyquasoare/c_quasoare_core.h"

/**
* Code to solve the unit inflow quadratic routing reservoir:
* dS/dt = 1 - S^2
*
* The reservoir has 2 flux functions:
* f1(S) = 1  (constant)
* f2(S) = -S^2
*
* The analytical solution of this reservoir is
* S(t) = [s0 + tanh(t)] / [1 + s0*tanh(t)]
*
* This model can be solved exactly with Quasoare.
*/


int main(){
    int nalphas = 4;
    double alphas[4] = {0., 0.4, 0.8, 1.2};

    /* Coefficient matrices : 2 fluxes x 3 interpolation bands x 3 coefs
     * containing 18 elements */
    int nfluxes = 2;
    double coefs[18];

    /* Interpolation coefficients */
    /* .. quadratic terms (only for f2) */
    fprintf(stdout, ".. Initialise coefficients\n");

    /* First flux : f1(S) = 1  for all bands */
    coefs[0] = 0; coefs[1] = 0; coefs[2] = 1;
    coefs[3] = 0; coefs[4] = 0; coefs[5] = 1;
    coefs[6] = 0; coefs[7] = 0; coefs[8] = 1;

    /* Second flux : f2(S) = -S^2  for all bands */
    coefs[9] = -1; coefs[10] = 0; coefs[11] = 0;
    coefs[12] = -1; coefs[13] = 0; coefs[14] = 0;
    coefs[15] = -1; coefs[16] = 0; coefs[17] = 0;

    /* Setup ODE */
    double s0 = 0.1;
    double t0 = 0.;
    double timestep = 0.01;
    double t1;
    int i, nval = 300;
    double scalings[2] = {1., 1.}; // no scaling here
    int niter[1];
    double s1[1];
    double fluxes[2];
    double anl, omega;

    /* Print header in result file */
    FILE *fp = fopen("example_quadroute.csv", "w");
    fprintf(fp, "time,store,inflow,outflow,store_analytical\n");

    /* integrate */
    fprintf(stdout, ".. Run model\n");
    for(i=0; i<nval; i++) {
        t1 = t0 + timestep * i;

        /* Store level and fluxes - QuaSoARe solution */
        c_quad_integrate(nalphas, nfluxes, alphas, scalings,
                         coefs, t0, s0, t1,
                         niter, s1, fluxes);

        /* Store level - Analytical solution */
        omega = tanh(t1);
        anl = (s0 + omega) / (1 + s0 * omega);

        fprintf(fp, "%0.8f,%0.8f,%0.8f,%0.8f,%0.8f\n",
                t1, s1[0], fluxes[0], fluxes[1], anl);
    }
    fclose(fp);
    fprintf(stdout, ".. process completed\n");
    return 0;
}

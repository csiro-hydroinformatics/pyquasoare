
#include <stdio.h>
#include "../src/pyrezeq/c_rezeq_utils.h"
#include "../src/pyrezeq/c_rezeq_quad.h"

int main(){
    int nalphas = 4;
    double alphas[4]={0., 0.4, 0.8, 1.2};

    /* Coefficient matrices : 4 alphas x 2 fluxes -> 3x2 matrices*/
    int nfluxes = 2;
    double amat[6], bmat[6], cmat[6];

    /* Quadratic reservoir ds/dt = 1-s^2 */
    amat[0] = 0; amat[1] = -1.;
    amat[2] = 0; amat[3] = -1.;
    amat[4] = 0; amat[5] = -1.;

    bmat[0] = 0; bmat[1] = 0;
    bmat[2] = 0; bmat[3] = 0;
    bmat[4] = 0; bmat[5] = 0;

    cmat[0] = 1; cmat[1] = 0;
    cmat[2] = 1; cmat[3] = 0;
    cmat[4] = 1; cmat[5] = 0;

    /* Setup ODE */
    double s0 = 0.1;
    double t0 = 0.;
    double timestep = 0.1;
    double t1;
    int i, nval = 10;
    double scalings[2] = {1., 1.}; // no scaling here
    int niter[1];
    double s1[1];
    double fluxes[2];

    /* integrate */
    for(i=0; i<nval; i++){
        t1 = t0+timestep*i;
        c_quad_integrate(nalphas, nfluxes, alphas, scalings,
                            amat, bmat, cmat, t0, s0, t1,
                            niter, s1, fluxes);
        fprintf(stdout, "[%3d] s1=%5.5e fx[0]=%5.5e fx[1]=%5.5e\n",
                    i, s1[0], fluxes[0], fluxes[1]);
    }

    return 0;
}
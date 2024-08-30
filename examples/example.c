
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
    double timestep = 0.01;
    double t1;
    int i, nval = 300;
    double scalings[2] = {1., 1.}; // no scaling here
    int niter[1];
    double s1[1];
    double fluxes[2];
    double anl, omega;

    /* Print header in result file */
    FILE *fp = fopen("example.csv", "w");
    fprintf(fp, "time,s1,flux1,flux2,s1_analytical\n");

    /* integrate */
    for(i=0; i<nval; i++){
        t1 = t0+timestep*i;
        omega = tanh(t1);
        anl = (s0+omega)/(1+s0*omega);
        c_quad_integrate(nalphas, nfluxes, alphas, scalings,
                            amat, bmat, cmat, t0, s0, t1,
                            niter, s1, fluxes);
        fprintf(fp, "%0.8f,%0.8f,%0.8f,%0.8f,%0.8f\n",
                    t1, s1[0], fluxes[0], fluxes[1], anl);
    }
    fclose(fp);

    return 0;
}

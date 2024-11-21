
#include <stdio.h>
#include "../../src/pyquasoare/c_quasoare_utils.h"
#include "../../src/pyquasoare/c_quasoare_core.h"

/**
* Code to solve the reservoir equation associated
* with the VIC soil moisture model as per Kavetski
* et al. (2006):
*
* dS/dt = P*(1-S/Smax)^alpha -Kb(S/Smax)^beta
*            -E[1-(1-S/Smax)^gamma]
*
* where P is rain and E is PET.
*
* In this implementation, the model is re-written as
*
* du/dt =   P/Smax
*            - P/Smax  * [1-(1-u)^alpha]
*            - Kb/Smax * u^beta
*            - E/Smax  * [1-(1-u)^gamma]
*
* where u = S/Smax
*
* Introducing the flux scaling factors, we finally obtain
*
* du/dt =  s1*f1(u)     -> f1(u) = 1 (constant)    s1=P/Smax
*          + s2*f2(u)   -> f2(u) = -1+(1-u)^alpha  s2=P/Smax
*          + s3*f3(u)   -> f3(u) = -u^beta         s3=Kb/Smax
*          + s4*f4(u)   -> f4(u) = -1+(1-u)^gamma  s4=E/Smax
*
* Kavetski, D., G. Kuczera, and S. W. Franks (2006),
* Bayesian analysis of input uncertainty in hydrological
* modeling: 2. Application, Water Resour. Res., 42,
* W03408, doi:10.1029/2005WR004376.
*/

/* Basic random number generation to simulate
 * climate inputs */
double rnd(){
    return (double)rand()/(double)RAND_MAX;
}


int main(){
    /* Set interpolation nodes - more refined close to
    * the edges to better represent non-linearities
    */
    int nalphas = 12;
    double alphas[12]={-0.01, 0, 0.03, 0.12, 0.25, 0.41,
                        0.59, 0.75, 0.88, 0.97, 1., 1.01};

    /* Coefficient matrices : 10 alphas x 4 fluxes
        -> 10x4 matrices*/
    int nfluxes = 4;
    double amat[48], bmat[48], cmat[48];

    /* Model parameters */
    double Smax = 100;
    double Kb = 1.;
    double alpha = 0.5;
    double beta = 2;
    double gamma = 10;

    /* Compute interpolation coefficients */
    fprintf(stdout, ".. Computing coefficients\n");
    int approx_opt = 1; /* Monotonous interpolation */
    int i, j;
    double a0, a1, am, f0, f1, fm;
    double coefs[3];

    for (i=0; i<nalphas-1; i++){
        a0 = alphas[i];
        a1 = alphas[i+1];
        am = (a0+a1)/2;

        for(j=0; j<nfluxes; j++){
            if(j==0){
                // Rainfall
                f0 = 1;
                f1 = 1;
                fm = 1;
            }
            else if (j==1){
                /* Effective rainfall */
                f0 = a0<0 ? 0 : -1+pow(1-a0, alpha);
                f1 = a1<0 ? 0 : -1+pow(1-a1, alpha);
                fm = am<0 ? 0 : -1+pow(1-am, alpha);
            }
            else if (j==2){
                /* Baseflow */
                f0 = a0<0 ? 0 : -pow(a0, beta);
                f1 = a1<0 ? 0 : -pow(a1, beta);
                fm = am<0 ? 0 : -pow(am, beta);
            }
            else if (j==3){
                /* Actual ET */
                f0 = a0 < 0 ? 0 : -1+pow(a0, gamma);
                f1 = a1 < 0 ? 0 : -1+pow(a1, gamma);
                fm = am < 0 ? 0 : -1+pow(am, gamma);
            }

            /* Compute coefficients */
            c_quad_coefficients(approx_opt, a0, a1, f0, f1, fm, coefs);

            /* Store coefficients in matrices */
            amat[i*nfluxes+j] = coefs[0];
            bmat[i*nfluxes+j] = coefs[1];
            cmat[i*nfluxes+j] = coefs[2];
        }
    }

    /* Setup ODE */
    double Pzero=0.9, w, P, E;
    double s0 = 0.5; /* Initial condition */
    double timestep = 1.; /* Daily simulation */
    int ndays = 3650; /* Assume a daily simulation of 10 years */
    double scalings[4];
    int niter[1];
    double s1[1];
    double fluxes[4];

    /* Print header in result file */
    FILE *fp = fopen("example_vicmod.csv", "w");
    fprintf(fp, "time,niter,P,E,S,Peff,Bflow,AET\n");

    /* integrate */
    fprintf(stdout, ".. Starting simulation\n");
    for(i=0; i<ndays; i++){
        if(i%200==0)
            fprintf(stdout, ".. Timestep %4d/%4d\n", i+1, ndays);

        /* Generate rain and PET values - replace with obs data ! */
        w = rnd();
        P =  w>Pzero ? -log(rnd())*5 : 0; // Exponential distrib with 5 mm and 90% zero
        E =  sin((double)i/365*3.14159)*2+3; // Seasonally varying PET with between 3 and 5 mm

        /* Set flux scaling factors */
        scalings[0] = P/Smax;
        scalings[1] = P/Smax;
        scalings[2] = Kb/Smax;
        scalings[3] = E/Smax;

        /* Integrate over timestep */
        c_quad_integrate(nalphas, nfluxes, alphas, scalings,
                            amat, bmat, cmat, 0, s0, timestep,
                            niter, s1, fluxes);

        /* Store */
        fprintf(fp, "%d,%d,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f\n",
                    i, niter[0], P, E, Smax*s1[0], -Smax*fluxes[1],
                    -Smax*fluxes[2], -Smax*fluxes[3]);

        /* Loop */
        s0 = s1[0];
    }
    fclose(fp);
    fprintf(stdout, ".. process completed\n");
    return 0;
}

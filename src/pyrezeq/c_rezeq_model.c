#include "c_rezeq_model.h"

/**
* Integrate reservoir equation over multiple time steps:
* - scalings [nval, nfluxes] : scalings applied to linear coefficients
* - other input args identical to c_integrate
* - s1 [nval] : final states
* - fluxes [nval, nfluxes] : flux computed
**/
int c_model(int nalphas, int nfluxes, int nval, double delta,
                            double * alphas,
                            double * scalings,
                            double nu,
                            double * a_matrix_noscaling,
                            double * b_matrix_noscaling,
                            double * c_matrix_noscaling,
                            double s0,
                            int * niter,
                            double * s1,
                            double * fluxes) {
    int ierr, t;
    double t0=0;

    for(t=0; t<nval; t++){
        ierr = c_integrate(nalphas, nfluxes,
                            alphas,
                            &(scalings[nfluxes*t]),
                            nu,
                            a_matrix_noscaling,
                            b_matrix_noscaling,
                            c_matrix_noscaling,
                            t0,
                            s0,
                            delta,
                            &(niter[t]),
                            &(s1[t]),
                            &(fluxes[nfluxes*t]));
        if(ierr>0)
            return ierr;

        /* Loop initial state */
        s0 = s1[t];
    }

    return 0;
}

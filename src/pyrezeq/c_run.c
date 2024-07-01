#include "c_run.h"

/**
* Integrate reservoir equation over multiple time steps:
* - scalings [nval, nfluxes] : scalings applied to linear coefficients
* - other input args identical to c_integrate
* - s1 [nval] : final states
* - fluxes [nval, nfluxes] : flux computed
**/
int c_run(int nalphas, int nfluxes, int nval, double delta,
                            double * alphas,
                            double * scalings,
                            double * nu_vector,
                            double * a_matrix_noscaling,
                            double * b_matrix_noscaling,
                            double * c_matrix_noscaling,
                            double s0,
                            double * s1,
                            double * fluxes) {
    int ierr, t;

    for(t=0; t<nval; t++){
        ierr = c_integrate(nalphas, nfluxes, delta,
                            alphas,
                            &(scalings[nfluxes*t]),
                            nu_vector,
                            a_matrix_noscaling,
                            b_matrix_noscaling,
                            c_matrix_noscaling,
                            s0,
                            &(s1[t]),
                            &(fluxes[nfluxes*t]));
        if(ierr>0)
            return ierr;

        /* Loop initial state */
        s0 = s1[t];
    }

    return 0;
}
#include "c_nonlinrouting.h"

int c_nonlinrouting(int nval, int nsubdiv, double delta,
                        double theta, double nu, double q0,
                        double s0, double *inflows, double * outflows){
    int i, j;
    double linear_thresh = 1+1e-10;
    double dt = 1./(double)nsubdiv;
    double s1, qi, qout;
    double qi_prev = 0;

    if(nsubdiv<1 || nsubdiv>1000)
        return NONLINROUT_ERROR + __LINE__;

    if(theta<1e-5)
        return NONLINROUT_ERROR + __LINE__;

    /* Cannot have nu<1 otherwise function is not
     * Lipschitz continuous */
    if(nu<1 || nu>10)
        return NONLINROUT_ERROR + __LINE__;

    if(q0<1e-5)
        return NONLINROUT_ERROR + __LINE__;

    if(s0<0 || s0>1e1*theta)
        return NONLINROUT_ERROR + __LINE__;

    /* Time series loop */
    for(i=0; i<nval; i++){
        qi = inflows[i];
        qi = isnan(qi) || qi<0 ? qi_prev : qi;

        /* Integrate */
        for(j=0; j<nsubdiv; j++){
            /* First operator : fixed inflow */
            s1 = s0+qi*dt;

            /* Second operator : integrate ds/dt = -q0(s/theta)^nu */
            if(nu>linear_thresh) {
                /* Non linear reservoir
                 * s^(-nu) ds = -q0/theta^nu.dt
                 * [1/(1-nu).s^(1-nu)] = -q0/theta^nu.Delta
                 * s1^(1-nu)=s0^(1-nu)-q0/theta^nu.Delta.(1-nu)
                 * s1 = [s0^(1-nu)+q0/theta^nu.Delta.(nu-1)]^(1/(1-nu))
                 */
                s1 = pow(pow(s1, 1-nu)+q0*pow(theta, nu)*(nu-1), 1./(1-nu));
            } else {
                /* Linear reservoir
                 * ds/dt = -q0(s/theta)
                 * ds/s = -q0/theta.dt
                 * log(s) = -q0/theta.Delta
                 * s1 = s0.exp(-q0/theta.Delta)
                 * */
                s1 = s1*exp(-q0/theta*delta);
            }
        }

        /* outflow computed from mass balance */
        qout = qi+(s0-s1)/delta;
        outflows[i] = qout;

        //fprintf(stdout, "[%4d] theta=%3.3e nu=%3.3e qi=%3.3e s0=%3.3e "
        //                    "-> s1=%3.3e qo=%3.3e\n",
        //                        i, theta, nu, qi, s0, s1, qout);

        /* Loop */
        s0 = s1;
        qi_prev = qi;
    }

    return 0;
}



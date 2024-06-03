#include "c_quadrouting.h"

int c_quadrouting(int nval, double delta, double theta, double q0,
                        double s0, double *inflow, double * outflow){
    int i;
    double s1, t, gamma, qi, qi_prev, qout, C;

    if(s0<0 || s0>theta-1e-5)
        return QUADROUT_ERROR + __LINE__;

    if(q0<1e-5)
        return QUADROUT_ERROR + __LINE__;

    if(theta<1e-5)
        return QUADROUT_ERROR + __LINE__;

    for(i=0; i<nval; i++){
        qi = inflow[i];
        qi = isnan(qi) ? qi_prev : qi;
        qi = qi>=0 ? qi : 0;

        gamma = theta*sqrt(qi/q0);

        /* Integrate */
        if(gamma>1e-10) {
            t = tanh(qi*delta/gamma);
            s1 = (s0+gamma*t)/(1+s0/gamma*t);
        }
        else {
            s1 = s0/(1+s0*delta*theta*theta/q0);
        }

        /* outflow computed from mass balance */
        qout = qi+(s0-s1)/delta;
        outflow[i] = qout;

        /* Loop */
        s0 = s1;
        qi_prev = qi;
    }

    return 0;
}



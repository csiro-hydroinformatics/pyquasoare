#include "c_gr4jprod.h"

int c_gr4jprod(int nval, int nsubdiv, double X1,
                        double s0,
                        double *inputs,
                        double *outputs){
    int i, j;
    double dt = 1./(double)nsubdiv;
    double S, SR, TWS, Pi, Ei, PR, PERC, S2;
    double P, E, PSi, ESi;
    double AE;

    /* Check input data */
    if(nsubdiv<1 || nsubdiv>1000)
        return GR4JPROD_ERROR + __LINE__;

    if(s0<1e-5 || s0>X1-1e-5)
        return GR4JPROD_ERROR + __LINE__;
    S = s0;

    if(X1<5.||X1>10000.)
        return GR4JPROD_ERROR + __LINE__;

    /* Time series loop */
    for(i=0; i<nval; i++){
        P = inputs[2*i];
        E = inputs[2*i+1];

        /* Interception */
        Pi = P>E ? (P-E)*dt: 0;
        Ei = E>P ? (E-P)*dt: 0;

        /* Initialise */
        AE = 0;
        PR = 0;
        PERC = 0;

        for(j=0; j<nsubdiv; j++){
            /* Effective rainfall */
            SR = S/X1;
            TWS = tanh(Pi/X1*dt);
            PSi = X1*(1-SR*SR)*TWS/(1+SR*TWS);
            PR += Pi-PSi;
            /* Actual ET */
            TWS = tanh(Ei/X1*dt);
            ESi = S*(2-SR)*TWS/(1+(1-SR)*TWS);
            AE += ESi+E-Ei;
            S += PSi-ESi;
            /* percolation */
            SR = S/X1/2.25;
            S2 = S/sqrt(sqrt(1.+SR*SR*SR*SR));
            PERC += S-S2;
            PR += S-S2;
            S = S2;
        }

        /* Store */
        outputs[4*i] = S;
        outputs[4*i+1] = PR;
        outputs[4*i+2] = AE;
        outputs[4*i+3] = PERC;
    }

    return 0;
}



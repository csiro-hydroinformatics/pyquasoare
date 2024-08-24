#include "c_gr4jprod.h"

int c_gr4jprod(int nval, int nsubdiv, double X1,
                        double s0,
                        double *inputs,
                        double *outputs){
    int i, j;
    double S, SR, PHI, PSI, Pi, Ei, PS, ES, PERC, S2;
    double P, E, PSi, ESi, PR, AE;

    double dt = 1./(double)nsubdiv;

    /* Check input data */
    if(nsubdiv<1 || nsubdiv>100000)
        return REZEQ_BENCH_NSUBDIV_TOO_HIGH;

    if(X1<5.||X1>10000.)
        return REZEQ_BENCH_PARAMS_OUT_OF_BOUNBDS;

    if(s0<1e-5 || s0>X1-1e-5)
        return REZEQ_BENCH_INITIALISATION_OUT_OF_BOUNBDS;
    S = s0;

    /* Time series loop */
    for(i=0; i<nval; i++){
        P = inputs[2*i];
        E = inputs[2*i+1];

        /* Interception */
        Pi = P>E ? (P-E)*dt: 0;
        Ei = E>P ? (E-P)*dt: 0;

        /* Initialise */
        PS = 0;
        ES = 0;
        PERC = 0;
        PR = 0;
        AE = 0;

        /* integrate equations of sub step dt */
        for(j=0; j<nsubdiv; j++){
            /* Effective rainfall */
            SR = S/X1;
            PHI = tanh(Pi/X1*dt);
            PSi = X1*(1-SR*SR)*PHI/(1+SR*PHI)/dt;
            PS += PSi;
            PR += Pi-PSi;

            /* Actual ET assuming SR is same because
             * either Pi is 0 or Ei is 0
             * */
            PSI = tanh(Ei/X1*dt);
            ESi = S*(2-SR)*PSI/(1+(1-SR)*PSI)/dt;
            ES += ESi;
            AE += ESi+P*dt-Pi;

            /* balance so far */
            S += PSi-ESi;

            /* percolation */
            SR = S/X1/2.25;
            S2 = S/sqrt(sqrt(1.+SR*SR*SR*SR*dt));
            PERC += S-S2;
            PR += S-S2;
            S = S2;
        }

        /* Store normalised fluxes to match with quad */
        outputs[6*i] = S;
        outputs[6*i+1] = PS;
        outputs[6*i+2] = ES;
        outputs[6*i+3] = PERC;
        outputs[6*i+4] = PR;
        outputs[6*i+5] = AE;
    }

    return 0;
}



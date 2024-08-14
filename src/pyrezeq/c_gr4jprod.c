#include "c_gr4jprod.h"

int c_gr4jprod(int nval, int nsubdiv, double X1,
                        double s0,
                        double *inputs,
                        double *outputs){
    int i, j;
    double S, SR, PHI, PSI, Pi, Ei, PS, ES, PERC, S2;
    double P, E, PSi, ESi;

    double dt = 1./(double)nsubdiv;

    /* Check input data */
    if(nsubdiv<1 || nsubdiv>10000)
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
        PS = 0;
        ES = 0;
        PERC = 0;

        /* integrate equations of sub step dt */
        for(j=0; j<nsubdiv; j++){
            /* Effective rainfall */
            SR = S/X1;
            PHI = tanh(Pi/X1*dt);
            PSi = X1*(1-SR*SR)*PHI/(1+SR*PHI)/dt;
            PS += PSi;
            /* We do not calculate PR because it requires
            * adding interception residual later. The following code can be
            * used:
            PR += Pi-PSi;
            * */

            /* Actual ET assuming SR is same because
             * either Pi is 0 or Ei is 0
             * */
            PSI = tanh(Ei/X1*dt);
            ESi = S*(2-SR)*PSI/(1+(1-SR)*PSI)/dt;
            ES += ESi;
            /* We do not calculate actual ET (AE) because it requires
            * adding interception residual later. The following can be
            * used:
            AE += ESi+P*dt-Pi;
            * */

            /* balance so far */
            S += PSi-ESi;

            /* percolation */
            SR = S/X1/2.25;
            S2 = S/sqrt(sqrt(1.+SR*SR*SR*SR*dt));
            PERC += S-S2;
            //PR += S-S2;
            S = S2;
        }

        /* The following code can be used to compute PR and AE:
        PR = P+(AE-PR)*X1+Ei*X1-E;
        AE += P-Pi*X1;
         */

        /* Store */
        outputs[4*i] = S;
        outputs[4*i+1] = PS;
        outputs[4*i+2] = ES;
        outputs[4*i+3] = PERC;
    }

    return 0;
}



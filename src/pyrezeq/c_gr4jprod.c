#include "c_gr4jprod.h"

int c_gr4jprod(int nval, int nsubdiv,double delta, double X1,
                        double s0,
                        double *rain, double *pet,
                        double * storage, double *actual_et, double * effective_rain){
    int i, j;
    double SR, TWS, WS, Pi, Ei, PS, ES, EN=0, PR, PERC, S2;
    double AE;

    if(nsubdiv>=1 || nsubdiv<=1000)
        return GR4JPROD_ERROR + __LINE__;

    if(s0<1e-5 || s0>X1-1e-5)
        return GR4JPROD_ERROR + __LINE__;

    if(X1<1e-5||X1>50000)
        return GR4JPROD_ERROR + __LINE__;

    SR = S/X1;


    for(i=0; i<nval; i++){
        /* Interception */
        Pi = P>E ? P-E : 0;
        Ei = E>P ? E-P : 0;

        PERC = 0;
        PR = 0;

        /* TODO : integrate that the integration is over delta/nsubdiv !!! */
        for(j=0; j<nsubdiv; j++){
            if(Pi>E)
            {
                WS =(P-E)/Scapacity;
                TWS = tanh(WS);

                ES = 0;
                PS = Scapacity*(1-SR*SR)*TWS;
                PS /= (1+SR*TWS);
            	PR = P-E-PS;
            	EN = 0;
                AE = E;
            }
            else
            {
            	WS = (E-P)/Scapacity;
                TWS = tanh(WS);

            	ES = S*(2-SR)*TWS;
                ES /= (1+(1-SR)*TWS);
            	PS = 0;
            	PR = 0;
            	EN = E-P;
                AE = ES+P;
            }
            S += PS-ES;

            /* percolation */
            SR = S/Scapacity/2.25;
            S2 = S/sqrt(sqrt(1.+SR*SR*SR*SR));

            PERC = S-S2;
            S = S2;
            PR += PERC;
        }
    }

    return 0;
}



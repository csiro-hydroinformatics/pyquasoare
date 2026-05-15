#include "c_quadgrad.h"

int c_quadgrad_runtimestep(double qstar1, double timestep, double gamma,
                           double x0, double * state, double * flux) {

    double delta = sqrt(qstar1 * timestep / gamma);
    double tau = tanh(delta);

    double x1 = (x0 + delta * tau) / (1. + x0 / delta * tau);

    *state = x1;
    *flux = qstar1 - gamma / timestep * (x1 - x0);

    return 0;
}


int c_quadgrad_run(int nval, int nreaches,
                   double timestep,
                   double x0,
                   double * gammas,
                   double * qin,
                   double * states,
                   double * qout) {
    int i, j, k;
    double qi;
    double xini;
    double gam;

    /* Loop through reaches */
    for(i = 0; i < nreaches; i++) {

        /* initialise reach model */
        xini = x0;
        gam = gammas[i];

        /* Time series loop */
        for(j = 0; j < nval; j++) {

            /* Get inflow */
            k = j * nreaches + i - 1;
            qi = i == 0 ? qin[j] : qout[k];

            /* Run model */
            k = j * nreaches + i;
            c_quadgrad_runtimestep(qi, timestep,
                                   gam, xini,
                                   &(states[k]),
                                   &(qout[k]));

            /* Set timestep for next iteration */
            xini = states[k];
        }
    }
    return 0;
}


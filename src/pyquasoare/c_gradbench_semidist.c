#include "c_gradbench_semidist.h"

int c_gradbench_semidist(int nval, int nreaches,
                         double timestep,
                         double x0_rr_prod,
                         double x0_rr_route,
                         double x0_ro_route,
                         double * areas,
                         double * gammas_rr_prod,
                         double * gammas_rr_route,
                         double * gammas_ro_route,
                         double * demands_rr_route,
                         double * pin,
                         double * ein,
                         double * states_rr_prod,
                         double * states_rr_route,
                         double * states_ro_route,
                         double * qout_rr_prod,
                         double * qout_rr_route,
                         double * qout_ro_route){

    int i, j, k;
    double qi, xini_rr_prod, xini_rr_route, xini_ro_route;
    double qout[1], conversion_factor;

    /* Loop through reaches */
    for(i = 0; i < nreaches; i++) {

        xini_rr_prod = x0_rr_prod;
        xini_rr_route = x0_rr_route;
        xini_ro_route = x0_ro_route;

        /* Convert mm/timestep[days] -> m3/s */
        conversion_factor = areas[i] * timestep / 86.4;

        /* Time series loop */
        for(j = 0; j < nval; j++) {
            k = j * nreaches + i;

            /* Rainfall-runoff modelling */
            c_gradbench_rainfall_runoff(1, timestep,
                                        xini_rr_prod,
                                        xini_rr_route,
                                        gammas_rr_prod[i],
                                        gammas_rr_route[i],
                                        demands_rr_route[i],
                                        &(pin[k]),
                                        &(ein[k]),
                                        &(states_rr_prod[k]),
                                        &(states_rr_route[k]),
                                        &(qout_rr_prod[k]),
                                        &(qout_rr_route[k]));

            xini_rr_prod = states_rr_prod[k];
            xini_rr_route = states_rr_route[k];

            /* Routing model - the first reach is runoff only */
            qi = i == 0 ? 0. : qout_ro_route[k - 1];
            ro_route_runtimestep(qi, timestep,
                                 gammas_ro_route[i],
                                 xini_ro_route,
                                 &(states_ro_route[k]),
                                 qout);

            xini_ro_route = states_ro_route[k];
            qout_ro_route[k] = *qout + qout_rr_route[k] * conversion_factor;
        }
    }

    return 0;
}

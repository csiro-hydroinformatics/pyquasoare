#include "c_gradbench_linrouting.h"


int rol_route_runtimestep(double qstar1, double timestep, double gamma,
                          double theta, double x0, double * state, double * flux) {
    /* Quasaore coefficients
     * dx/dt = Q1* / Γ - 1/Δ f(x)
     * f(x) = 1/θ x    with θ dimless
     * hence dx/dt = Q1* / Γ - θ/Δ  x
     *
     * x(t) = Q1* Δ * θ / Γ + (x0 - Q1* Δ * θ / Γ ) exp(- x / θΔ)
     *
     * Output flux = Q1* - Q1 - Q2
     * F = Q1* - Q1* (1 - intx / Δ)
     */

    double xbar = qstar1 * timestep * theta / gamma ;
    double x1 = xbar + (x0 - xbar) * exp(-1./theta);

    *state = x1;
    *flux = qstar1 - gamma / timestep * (x1 - x0);

    return 0;
}

int c_gradbench_linrouting(int nval, int nreaches,
                           double timestep,
                           double * x0_ro_route,
                           double * thetas_ro_route,
                           double * qin,
                           double * states_ro_route,
                           double * qout_ro_route) {
    int i, j, k;
    double qi;
    double xini;
    double theta;

    /* Loop through reaches */
    for(i = 0; i < nreaches; i++) {

        /* initialise reach model */
        xini = x0_ro_route[i];
        theta = thetas_ro_route[i];

        /* Time series loop */
        for(j = 0; j < nval; j++) {

            /* Get inflow */
            k = j * nreaches + i - 1;
            qi = i == 0 ? qin[j] : qout_ro_route[k];

            /* Run model */
            k = j * nreaches + i;
            rol_route_runtimestep(qi, timestep, 1., theta, xini,
                                  &(states_ro_route[k]),
                                  &(qout_ro_route[k]));

            /* Set timestep for next iteration */
            xini = states_ro_route[k];
        }
    }
    return 0;
}


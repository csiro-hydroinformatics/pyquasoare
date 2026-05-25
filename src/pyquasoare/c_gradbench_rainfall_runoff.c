#include "c_gradbench_rainfall_runoff.h"

int rr_prod_runtimestep(double qstar1, double qstar2,
                       double timestep, double gamma,
                       double x0, double * state, double * flux) {
    /* Quasaore coefficients
     * dx/dt = Q1* / G (1-x^2) - Q2* / G x(2-x) - 1/Δ  4^3/9^4 x^2
     *       = a x^2 + b x + c
     *
     * Output flux = Q1* - Q1 - Q2
     * F = Q1* - Q1* (1 - intx2/Δ) + Γ/Δ^2 4^3/9^4 intx2
     */
    double a = (qstar2 - qstar1) / gamma - 64. / 6561. / timestep;
    double b = -2 * qstar2 / gamma;
    double c = qstar1 / gamma;
    double constants[4];
    gb_constants(a, b, c, constants);

    /* Applies Quasaore solution */
    double x1 = gb_forward(a, b, c, constants, timestep, x0);

    *state = x1;

    /* compute output flux, i.e. Q1*-Q1 + Q3 */
    double integ[2];
    gb_integrate_x(a, b, c, constants, timestep, x0, x1, integ);
    *flux = (qstar1 + gamma * 64. / 6561. / timestep) * integ[1] / timestep;

    return 0;
}

int rr_route_runtimestep(double qstar1, double qstar2,
                        double timestep, double gamma,
                        double x0, double * state, double * flux) {
    /* Quasaore coefficients
     * dx/dt = Q1* / G 0.9 + Q2* / G x^2 - 1/Δ 0.25 x^2
     *
     * Output flux = Q1* - Q1 - Q3
     * F = 0.1 Q1* + Γ/Δ 0.25 intx2 / Δ
     *
     */
    double a = qstar2 / gamma - 0.25 / timestep;
    double b = 0.;
    double c = qstar1 / gamma * 0.9;
    double constants[4];
    gb_constants(a, b, c, constants);

    /* Applies Quasaore solution */
    double x1 = gb_forward(a, b, c, constants, timestep, x0);

    *state = x1;

    double integ[2];
    gb_integrate_x(a, b, c, constants, timestep, x0, x1, integ);
    *flux = qstar1 * 0.1 + gamma * 0.25 * integ[1] / timestep / timestep;

    return 0;
}

int c_gradbench_rainfall_runoff(int nval,
                                double timestep,
                                double x0_rr_prod,
                                double x0_rr_route,
                                double gamma_rr_prod,
                                double gamma_rr_route,
                                double demand_rr_route,
                                double * pin,
                                double * ein,
                                double * states_rr_prod,
                                double * states_rr_route,
                                double * qout_rr_prod,
                                double * qout_rr_route){
    int j;
    double pi, ei, qi;

    /* Time series loop */
    for(j = 0; j < nval; j++) {

        /* Get inputs for production */
        pi = pin[j];
        ei = ein[j];

        /* Run production */
        rr_prod_runtimestep(pi, ei, timestep, gamma_rr_prod,
                            x0_rr_prod,
                            &(states_rr_prod[j]),
                            &(qout_rr_prod[j]));

        /* Run routing */
        qi = qout_rr_prod[j];
        rr_route_runtimestep(qi, demand_rr_route,
                             timestep, gamma_rr_route,
                             x0_rr_route,
                             &(states_rr_route[j]),
                             &(qout_rr_route[j]));

        /* Set timestep for next iteration */
        x0_rr_prod = states_rr_prod[j];
        x0_rr_route = states_rr_route[j];
    }

    return 0;
}

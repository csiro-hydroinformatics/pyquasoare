#include "c_gradbench_semidist.h"

/* Functions copied from quasaore_core to simplify application of Tapenade */
int gb_sd_notnull(double x){
    return (x < 0 || x > 0 ? 1 : 0);
}

int gb_sd_isnull(double x){
    return 1 - gb_sd_notnull(x);
}

double gb_sd_omega_fun(double x, double Delta){
    if(Delta<0.)
        return tan(x);
    else
        return tanh(x);
}

int gb_sd_constants(double a, double b, double c, double constants[4]) {

    /* Delta (not very robust numerically, see c_quasoare_core for
     * a better formulation */
    double Delta = b * b - 4 * a * c;
    constants[0] = Delta;

    /* signD */
    constants[1] = Delta < 0 ? -1. : 1.;

    /* qD */
    constants[2] = sqrt(fabs(Delta)) / 2;

    /* sbar */
    constants[3] = -b / 2. / a;

    return 0;
}

double gb_sd_forward(double a, double b, double c, double constants[4],
                  double timestep, double s0){
    double s1 = 0;
    double omega = 0.;
    double exparg = 0;
    double tau = timestep;

    double Delta = constants[0];
    double signD = constants[1];
    double qD = constants[2];
    double sbar = constants[3];

    if(gb_sd_isnull(a) && gb_sd_isnull(b)){
        s1 = s0 + c * tau;
    }
    else if((gb_sd_isnull(a) && gb_sd_notnull(b))){
        exparg = b * tau;
        s1 = fabs(exparg) < 1e-12 ? s0 * (1 + exparg) + c * tau :
                -c / b + (s0 + c / b) * exp(exparg);
    }
    else{
        if(gb_sd_isnull(Delta)){
            s1 = sbar + (s0 - sbar) / (1 - a * tau * (s0 - sbar));
        }
        else {
            omega = gb_sd_omega_fun(qD*tau, Delta);
            s1 = sbar + (s0 - sbar - signD * qD / a * omega)
                        / (1. - a / qD * (s0 - sbar) * omega);
        }
    }
    return s1;
}

/* computes integral of x and x^2 over [0, Δ] */
double gb_sd_integrate_x(double a, double b, double c, double constants[4],
                      double timestep, double s0, double s1,
                      double integ[2])  {
    double tau = timestep;
    double tau2 = tau * tau;
    double tau3 = tau2 * tau;

    double Delta = constants[0];
    double signD = constants[1];
    double qD = constants[2];
    double sbar = constants[3];

    double integS, integS2, omega, term1, term2;

    if (gb_sd_isnull(a) && gb_sd_isnull(b)) {
        integS = s0 * tau + c * tau2 / 2;
        integS2 = s0 * s0 * tau + s0 * c * tau2 + c * c * tau3 / 3.;
    }
    else if((gb_sd_isnull(a) && gb_sd_notnull(b))){
        integS = (s1 - s0 - c * tau) / b;
        integS2 = ((s1 * s1 - s0 * s0) / 2. - c * integS) / b;
    }
    else if (gb_sd_notnull(a)){
        /* Integrate S only because S2 can be deducted from S */
        if(gb_sd_isnull(Delta))
            integS = sbar * tau - log(1 - a * tau * (s0 - sbar)) / a;
        else {
            omega = gb_sd_omega_fun(qD * tau, Delta);

            /* Special formula to avoid the case where omega == 1 */
            if(qD * tau > 10 && Delta > 0)
                term1 = (log(2) - qD * tau) / a;
            else
                term1 = log(1 - signD * omega * omega) / 2. / a;

            term2 = -log(1 - a * (s0 - sbar) / qD * omega) / a;
            integS = sbar * tau + term1 + term2;
        }

        /* Eq B13 in Lerat (2025) */
        integS2 = (s1 - s0 - b * integS - c * tau) / a;
    }

    integ[0] = integS;
    integ[1] = integS2;
}


int sd_rr_prod_runtimestep(double qstar1, double qstar2,
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
    gb_sd_constants(a, b, c, constants);

    /* Applies Quasaore solution */
    double x1 = gb_sd_forward(a, b, c, constants, timestep, x0);

    *state = x1;

    /* compute output flux, i.e. Q1*-Q1 + Q3 */
    double integ[2];
    gb_sd_integrate_x(a, b, c, constants, timestep, x0, x1, integ);
    *flux = (qstar1 + gamma * 64. / 6561. / timestep) * integ[1] / timestep;

    return 0;
}

int sd_rr_route_runtimestep(double qstar1, double qstar2,
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
    gb_sd_constants(a, b, c, constants);

    /* Applies Quasaore solution */
    double x1 = gb_sd_forward(a, b, c, constants, timestep, x0);

    *state = x1;

    double integ[2];
    gb_sd_integrate_x(a, b, c, constants, timestep, x0, x1, integ);
    *flux = qstar1 * 0.1 + gamma * 0.25 * integ[1] / timestep / timestep;

    return 0;
}

int sd_ro_route_runtimestep(double qstar1, double timestep, double gamma,
                            double x0, double * state, double * flux) {
    /* Quasaore coefficients
     * dx/dt = Q1* / Γ - 1/Δ x^2
     *       = a x^2 + b x + c
     *
     * Output flux = Q1* - Q1 - Q2
     * F = Q1* - Q1* (1 - intx2) + Γ/Δ  4^3/9^4 intx2
     */

    double a = -1. / timestep;
    double b = 0;
    double c = qstar1 / gamma;
    double constants[4];
    gb_sd_constants(a, b, c, constants);

    /* Applies Quasaore solution */
    double x1 = gb_sd_forward(a, b, c, constants, timestep, x0);

    *state = x1;
    *flux = qstar1 - gamma / timestep * (x1 - x0);

    return 0;
}


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
    double pi, ei, qi;
    double xini_rr_prod, xini_rr_route, xini_ro_route;
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

            /* Get inputs for production */
            pi = pin[k];
            ei = ein[k];

            /* Run production */
            sd_rr_prod_runtimestep(pi, ei, timestep, gammas_rr_prod[i],
                                   xini_rr_prod,
                                   &(states_rr_prod[k]),
                                   &(qout_rr_prod[k]));
            xini_rr_prod = states_rr_prod[k];

            /* Run routing */
            qi = qout_rr_prod[k];
            sd_rr_route_runtimestep(qi, demands_rr_route[i],
                                    timestep, gammas_rr_route[i],
                                    xini_rr_route,
                                    &(states_rr_route[k]),
                                    &(qout_rr_route[k]));
            xini_rr_route = states_rr_route[k];

            /* Routing model - the first reach is runoff only */
            qi = i == 0 ? 0. : qout_ro_route[k - 1];
            sd_ro_route_runtimestep(qi, timestep,
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

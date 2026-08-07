#include "c_gradbench_rainfall_runoff.h"

/* Functions copied from quasaore_core to simplify application of Tapenade */
int gb_rr_notnull(double x){
     return x < 0 || x > 0 ? 1 : 0;
}

int gb_rr_isnull(double x){
    return 1 - gb_rr_notnull(x);
}

double gb_rr_omega_fun(double x, double Delta){
    if(Delta<0.)
        return tan(x);
    else
        return tanh(x);
}

double gb_rr_expmxy(double x, double y) {
    double z = x * y;
    return x * (expm1(z) / z);
}

double gb_rr_logpxy(double x, double y) {
    double z = x * y;
    return x * (log1p(-z) / z);
}


int gb_rr_constants(double a, double b, double c, double constants[4]) {

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

double gb_rr_forward(double a, double b, double c, double constants[4],
                  double timestep, double s0){
    double s1 = 0;
    double omega = 0.;
    double exparg = 0;
    double tau = timestep;

    double Delta = constants[0];
    double signD = constants[1];
    double qD = constants[2];
    double sbar = constants[3];
    double expmb;

    if(gb_rr_isnull(a) && gb_rr_isnull(b)){
        s1 = s0 + c * tau;
    }
    else if((gb_rr_isnull(a) && gb_rr_notnull(b))){
        exparg = b * tau;
        if(fabs(exparg) < 1e-12) {
            s1 = s0 * (1 + exparg) + c * tau;
        } else {
            expmb = gb_rr_expmxy(tau, b);
            s1 = s0 * exp(exparg) + expmb * c;
        }
    }
    else{
        if(gb_rr_isnull(Delta)){
            s1 = sbar + (s0 - sbar) / (1 - a * tau * (s0 - sbar));
        }
        else {
            omega = gb_rr_omega_fun(qD*tau, Delta);
            s1 = sbar + (s0 - sbar - signD * qD / a * omega)
                        / (1. - a / qD * (s0 - sbar) * omega);
        }
    }
    return s1;
}

double gb_rr_integrate_x(double a, double b, double c, double constants[4],
                      double timestep, double s0, double s1,
                      double integ[2])  {
    double tau = timestep;

    double Delta = constants[0];
    double signD = constants[1];
    double qD = constants[2];
    double sbar = constants[3];
    double smax = fabs(s0) < fabs(s1) ? fabs(s1) : fabs(s0);

    double integS, integS2, term0, term1, term2, term12;
    double c_b_ratio, expmb, expm2b, l0, l1, y0, y1;

    term0 = fabs(c);
    term1 = fabs(b * smax);
    term2 = fabs(a * smax * smax);
    term12 = term1 > term2 ? term1 : term2;

    if (term0 * 1e-5 > term12 || gb_rr_isnull(fabs(s1 - s0))) {
        integS = (s0 + c * tau / 2) * tau;
        integS2 = (s0 * s0 + s0 * c * tau + c * c * tau * tau / 3.) * tau;
    }
    else if (term1 * 1e-5 > term2) {
        c_b_ratio = c / b;
        expmb = gb_rr_expmxy(tau, b); // (exp(b * tau) - 1) / b
        integS = s0 * expmb + (expmb - tau) * c_b_ratio;

        expm2b = gb_rr_expmxy(tau, 2 * b);
        integS2 = s0 * s0 * expm2b
            + c_b_ratio * c_b_ratio * (tau + expm2b - 2 * expmb)
            + 2 * s0 * c_b_ratio * (expm2b - expmb);
    }
    else {
        if(fabs(Delta) < 1e-15) {
            integS = sbar * tau - gb_rr_logpxy(tau * (s0 - sbar), a);
            integS2 = (s1 - s0 - b * integS - c * tau) / a;
        }
        else {
            y0 = a * s0 + b / 2;
            l0 = log(fabs(y0 * y0 - Delta / 4));

            y1 = a * s1 + b / 2;
            l1 = log(fabs(y1 * y1 - Delta / 4));

            if(!isfinite(l1) || fabs(y1 * y1 - Delta) < 1e-15)
               l1 = log(Delta) + log(fabs((y0 + qD) / (y0 - qD))) - 2 * qD * tau;

            integS = (l1 - l0 - b * tau) / 2 / a;
            integS2 = (y1 - y0 + b * (l0 - l1) / 2 + (b * b + Delta) * tau / 4) / a / a;
        }
    }
    integ[0] = integS;
    integ[1] = integS2;
}

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
    gb_rr_constants(a, b, c, constants);

    /* Applies Quasaore solution */
    double x1 = gb_rr_forward(a, b, c, constants, timestep, x0);

    *state = x1;

    /* compute output flux, i.e. Q1*-Q1 + Q3 */
    double integ[2];
    gb_rr_integrate_x(a, b, c, constants, timestep, x0, x1, integ);
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
    gb_rr_constants(a, b, c, constants);

    /* Applies Quasaore solution */
    double x1 = gb_rr_forward(a, b, c, constants, timestep, x0);

    *state = x1;

    double integ[2];
    gb_rr_integrate_x(a, b, c, constants, timestep, x0, x1, integ);
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

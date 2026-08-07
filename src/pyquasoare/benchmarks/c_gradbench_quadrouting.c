#include "c_gradbench_quadrouting.h"

/* Functions copied from quasaore_core to simplify application of Tapenade */
int gb_ro_notnull(double x){
    return x < 0 || x > 0 ? 1 : 0;
}

int gb_ro_isnull(double x){
    return 1 - gb_ro_notnull(x);
}

double gb_ro_omega_fun(double x, double Delta){
    if(Delta<0.)
        return tan(x);
    else
        return tanh(x);
}

double gb_ro_expmxy(double x, double y) {
    double z = x * y;
    return x * (expm1(z) / z);
}

double gb_ro_logpxy(double x, double y) {
    double z = x * y;
    return x * (log1p(-z) / z);
}

int gb_ro_constants(double a, double b, double c, double constants[4]) {

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

double gb_ro_forward(double a, double b, double c, double constants[4],
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

    if(gb_ro_isnull(a) && gb_ro_isnull(b)){
        s1 = s0 + c * tau;
    }
    else if((gb_ro_isnull(a) && gb_ro_notnull(b))){
        exparg = b * tau;
        if(fabs(exparg) < 1e-12) {
            s1 = s0 * (1 + exparg) + c * tau;
        } else {
            expmb = gb_ro_expmxy(tau, b);
            s1 = s0 * exp(exparg) + expmb * c;
        }
    }
    else{
        if(gb_ro_isnull(Delta)){
            s1 = sbar + (s0 - sbar) / (1 - a * tau * (s0 - sbar));
        }
        else {
            omega = gb_ro_omega_fun(qD*tau, Delta);
            s1 = sbar + (s0 - sbar - signD * qD / a * omega)
                        / (1. - a / qD * (s0 - sbar) * omega);
        }
    }
    return s1;
}

/* computes integral of x and x^2 over [0, Δ] */
double gb_ro_integrate_x(double a, double b, double c, double constants[4],
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

    if (term0 * 1e-5 > term12 || gb_ro_isnull(fabs(s1 - s0))) {
        integS = (s0 + c * tau / 2) * tau;
        integS2 = (s0 * s0 + s0 * c * tau + c * c * tau * tau / 3.) * tau;
    }
    else if (term1 * 1e-5 > term2) {
        c_b_ratio = c / b;
        expmb = gb_ro_expmxy(tau, b); // (exp(b * tau) - 1) / b
        integS = s0 * expmb + (expmb - tau) * c_b_ratio;

        expm2b = gb_ro_expmxy(tau, 2 * b);
        integS2 = s0 * s0 * expm2b
            + c_b_ratio * c_b_ratio * (tau + expm2b - 2 * expmb)
            + 2 * s0 * c_b_ratio * (expm2b - expmb);
    }
    else {
        if(fabs(Delta) < 1e-15) {
            integS = sbar * tau - gb_ro_logpxy(tau * (s0 - sbar), a);
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


int ro_route_runtimestep(double qstar1, double timestep, double gamma,
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
    gb_ro_constants(a, b, c, constants);

    /* Applies Quasaore solution */
    double x1 = gb_ro_forward(a, b, c, constants, timestep, x0);

    *state = x1;
    *flux = qstar1 - gamma / timestep * (x1 - x0);

    return 0;
}

int c_gradbench_quadrouting(int nval, int nreaches,
                            double timestep,
                            double * x0_ro_route,
                            double * gammas_ro_route,
                            double * qin,
                            double * states_ro_route,
                            double * qout_ro_route) {
    int i, j, k;
    double qi;
    double xini;
    double gam;

    /* Loop through reaches */
    for(i = 0; i < nreaches; i++) {

        /* initialise reach model */
        xini = x0_ro_route[i];
        gam = gammas_ro_route[i];

        /* Time series loop */
        for(j = 0; j < nval; j++) {

            /* Get inflow */
            k = j * nreaches + i - 1;
            qi = i == 0 ? qin[j] : qout_ro_route[k];

            /* Run model */
            k = j * nreaches + i;
            ro_route_runtimestep(qi, timestep, gam, xini,
                                 &(states_ro_route[k]),
                                 &(qout_ro_route[k]));

            /* Set timestep for next iteration */
            xini = states_ro_route[k];
        }
    }
    return 0;
}


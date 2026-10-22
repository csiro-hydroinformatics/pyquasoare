#include "c_gradbench_quadrouting.h"

/* Functions copied from quasaore_core to simplify application of Tapenade */
int gb_ro_notnull(double x){
    return x < 0 || x > 0 ? 1 : 0;
}

int gb_ro_isnull(double x){
    return 1 - gb_ro_notnull(x);
}

double gb_ro_omega_fun(double x, double Delta){
    if(Delta < 0.)
        return tan(x);
    else
        return tanh(x);
}

double gb_ro_expmxy(double x, double y) {
    double z = x * y;
    if(gb_ro_isnull(y) || gb_ro_isnull(x))
        return x;
    // expm1 is not known to Tapenade. Revert to naive form
    return x * ((exp(z) - 1) / z);
}

double gb_ro_logpxy(double x, double y) {
    double z = x * y;
    if(gb_ro_isnull(y) || gb_ro_isnull(x))
        return x;
    // log1p is not known to Tapenade. Revert to naive form
    return x * (log(1 - z) / z);
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


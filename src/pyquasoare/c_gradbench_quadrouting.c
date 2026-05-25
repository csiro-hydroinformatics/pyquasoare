#include "c_gradbench_quadrouting.h"

/* Functions copied from quasaore_core to simplify application of Tapenade */
int gb_notnull(double x){
    return x < 0 || x > 0 ? 1 : 0;
}

int gb_isnull(double x){
    return 1 - gb_notnull(x);
}

double gb_omega_fun(double x, double Delta){
    if(Delta<0.)
        return tan(x);
    else
        return tanh(x);
}

int gb_constants(double a, double b, double c, double constants[4]) {

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

double gb_forward(double a, double b, double c, double constants[4],
                  double timestep, double s0){
    double s1 = 0;
    double omega = 0.;
    double exparg = 0;
    double tau = timestep;

    double Delta = constants[0];
    double signD = constants[1];
    double qD = constants[2];
    double sbar = constants[3];

    if(gb_isnull(a) && gb_isnull(b)){
        s1 = s0 + c * tau;
    }
    else if((gb_isnull(a) && gb_notnull(b))){
        exparg = b * tau;
        s1 = fabs(exparg) < 1e-12 ? s0 * (1 + exparg) + c * tau :
                -c / b + (s0 + c / b) * exp(exparg);
    }
    else{
        if(gb_isnull(Delta)){
            s1 = sbar + (s0 - sbar) / (1 - a * tau * (s0 - sbar));
        }
        else {
            omega = gb_omega_fun(qD*tau, Delta);
            s1 = sbar + (s0 - sbar - signD * qD / a * omega)
                        / (1. - a / qD * (s0 - sbar) * omega);
        }
    }
    return s1;
}

/* computes integral of x and x^2 over [0, Δ] */
double gb_integrate_x(double a, double b, double c, double constants[4],
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

    if (gb_isnull(a) && gb_isnull(b)) {
        integS = s0 * tau + c * tau2 / 2;
        integS2 = s0 * s0 * tau + s0 * c * tau2 + c * c * tau3 / 3.;
    }
    else if((gb_isnull(a) && gb_notnull(b))){
        integS = (s1 - s0 - c * tau) / b;
        integS2 = ((s1 * s1 - s0 * s0) / 2. - c * integS) / b;
    }
    else if (gb_notnull(a)){
        /* Integrate S only because S2 can be deducted from S */
        if(gb_isnull(Delta))
            integS = sbar * tau - log(1 - a * tau * (s0 - sbar)) / a;
        else {
            omega = gb_omega_fun(qD * tau, Delta);

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
    gb_constants(a, b, c, constants);

    /* Applies Quasaore solution */
    double x1 = gb_forward(a, b, c, constants, timestep, x0);

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


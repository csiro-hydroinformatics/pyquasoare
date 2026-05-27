#include "c_quasoare_grad.h"

/* Solution of dS/dt = f*(s) */
double c_lin_forward(double M, double S, double t0, double x0, double t){
    double x1 = c_get_nan();
    double tau = t - t0;
    double exparg;

    if(isequal(t, t0, QUASOARE_EPS, 0.))
        return x0;

    if(isnull(S)){
        x1 = x0 + M * tau;
    }
    else {
        exparg = S * tau;
        x1 = fabs(exparg) < QUASOARE_EPS ? x0 * (1 + exparg) + M * tau
            : -M / S +(x0 + M / S) * exp(exparg);
    }
    return x1;
}


/* Primitive of 1/f*(s) */
double c_lin_inverse(double M, double S, double x0, double x1){
    if(isnull(S)){
        return (x1 - x0) / M;
    }
    else {
        return 1./S * log(fabs((S * x1 + M) / (S * x0 + M)));
    }
}


double c_lin_get_qstar(double Qstars[2], double gamma, double timestep, int i) {
    return i < 2 ? Qstars[i] : -gamma / timestep;
}


/* Increment fluxes by integrating f*(x) */
int c_lin_fluxes_increment(double timestep,
                           double gamma,
                           double Qstars[2],
                           double mj_vector[3],
                           double sj_vector[3],
                           double M, double S,
                           double t0, double t1,
                           double x0, double x1,
                           double fluxes[6]){
    int i;
    int nfluxes_primary = 3;
    double tau = t1 - t0;
    double tau2 = tau * tau;
    double tau3 = tau2 * tau;
    double mij, sij;
    double integS = c_get_nan();
    double dflux;
    double qst;

    if(t1<t0)
        return QUASOARE_TIME_TOOLOW;

    if(isnull(tau))
        return 0;

    /* Calculate integral of S and S^2 */
    if (isnull(S))
        integS = x0 * tau + M * tau2 / 2;
    else
        integS = (x1 - x0 - M * tau) / S;

    /* increment fluxes */
    for(i = 0; i < nfluxes_primary; i++){
        mij = mj_vector[i];
        sij = sj_vector[i];

        qst = c_lin_get_qstar(Qstars, gamma, timestep, i);
        dflux = qst * (sij * integS + mij * tau) / timestep;
        fluxes[i] += dflux;

        if(i <= 1)
            fluxes[i + 3] += qst - dflux;
    }

    /* Total outflow = bypasss (counted negatively) + outflow */
    fluxes[5] = - fluxes[3] + fluxes[4];

    return 0;
}


/* Integrate reservoir equation over 1 time step and compute associated fluxes */
int c_lin_integrate(int nalphas,
                    double * alphas,
                    double timestep,
                    double gamma,
                    double Qstars[2],
                    double * coefs_linear,
                    double t0,
                    double x0,
                    int *niter,
                    double * x1,
                    double fluxes[6]) {

    int idx, i, nit=0, jalpha_next=0, err_flux;
    int nfluxes_primary = 3;
    int nfluxes = 6;
    int ncoefs = 2;

    double M = 0., S = 0.;
    double mij = 0., sij = 0.;
    double qi;
    double funval = 0.;
    double funval_prev = 0.;
    double alpha0, alpha1, scl;
    double alpha_min = alphas[0];
    double alpha_max = alphas[nalphas-1];

    double mj_vect[3];
    double sj_vect[3];

    /* Max number of iteration
     * If s0 stays in [alpha0, alpha1], this number
     * should be <=nalphas. However we allow more iterations
     * if there is extrapolation */
    int niter_max = nalphas < 5 ? 10 : 2 * nalphas;

    /* Initial band */
    int jalpha = c_find_alpha(nalphas, alphas, x0);

    /* Inialise variables */
    int extrapolating = 0;
    int extrapolating_low = 0;
    int extrapolating_high = 0;
    double x_low = 0;
    double x_high = 0;
    double t_final = t0 + timestep;
    double t_start = t0;
    double t_end = t0;
    double x_start = x0;
    double x_end = x0;

    for(i=0; i<nfluxes; i++)
        fluxes[i] = 0.;

    /* Time loop */
    while (t_end < t_final * (1 - QUASOARE_EPS) && nit < niter_max) {
        nit += 1;

        /* Extrapolation is triggered if s_start is
         * below alpha0+EPS for low and
         * above alpha1-EPS for high */
        extrapolating_low = jalpha < 0 ? 1 : 0;
        extrapolating_high = jalpha >= nalphas - 1 ? 1 : 0;
        extrapolating = extrapolating_low + extrapolating_high > 0 ? 1 : 0;

        /* Get band limits */
        alpha0 = extrapolating_low ? -1 * c_get_inf() : alphas[jalpha];
        alpha1 = extrapolating_high ? c_get_inf() : alphas[jalpha+1];

        /* Sum coefficients accross fluxes */
        M = 0.; S = 0.;

        for(i=0 ; i < nfluxes_primary ; i++) {
            /* Compute flux coefficients */

            if(extrapolating_low)
                idx = (nalphas - 1) * ncoefs * i;
            else if(extrapolating_high)
                idx = (nalphas - 1) * ncoefs * i + ncoefs * (nalphas - 2);
            else
                idx = (nalphas - 1) * ncoefs * i + ncoefs * jalpha;

            sij = coefs_linear[idx];        // Slope
            mij = coefs_linear[idx + 1];    // Intercept

            /* All flux functions are assumed positive */
            if(1 - extrapolating &&
                    (mij + sij * alpha0 < 0 || mij + sij * alpha1 < 0))
                return QUASOARE_NEGATIVE_FLUX_FUNCTION;

            /* Store flux coefficents */
            mj_vect[i] = mij;
            sj_vect[i] = sij;

            /* Add coefficients to obtain approximation of reservoir equation
             * function */
            qi = c_lin_get_qstar(Qstars, gamma, timestep, i) / gamma;
            M += mij * qi;
            S += sij * qi;
        }

        /* Check coefficients */
        if(isnan(M) || isnan(S))
            return QUASOARE_NAN_COEFF;

        M = fabs(M) < QUASOARE_EPS ? 0. : M;
        S = fabs(S) < QUASOARE_EPS ? 0. : S;

        /* Get derivative at beginning of time step */
        funval = M + S * x_start;

        /* Check continuity except for first iteration */
        if(nit > 1){
            if(notequal(funval_prev, funval, QUASOARE_ATOL, QUASOARE_RTOL)) {
                return QUASOARE_NOT_CONTINUOUS;
            }
        }

        /* Try integrating up to the end of the time step */
        if(fabs(funval) < QUASOARE_EPS)
            x_end = x_start;
        else
            x_end = c_lin_forward(M, S, t_start, x_start, t_final);

        /* complete or move band if needed */
        if(x_end >= alpha0 && x_end <= alpha1) {
            /* .. s_end is within band => complete */
            t_end = t_final;
            jalpha_next = jalpha;
        }
        else {
            /* find next band depending if f is decreasing or non-decreasing */
            /* Increment time */
            if(extrapolating) {
                /* Ensure that x_end remains inside interpolation range */
                x_low = alpha_min + 2 * QUASOARE_EPS;
                x_high = alpha_max - 2 * QUASOARE_EPS;
                if(funval < 0 && extrapolating_high && x_end < x_high) {
                    x_end = x_high;
                    jalpha_next = nalphas-2;
                }
                else if (funval > 0 && extrapolating_low && x_end > x_low) {
                    x_end = x_low;
                    jalpha_next = 0;
                }
            }
            else {
                /* funval cannot be zero here because otherwise s_end
                 * is in [alpha0, alpha1] */
                if(funval < 0) {
                    x_end = alpha0;
                    jalpha_next = jalpha > -1 ? jalpha - 1 : -1;
                }
                else {
                    x_end = alpha1;
                    jalpha_next = jalpha >= nalphas - 2 ? nalphas - 1 : jalpha + 1;
                }
            }

            /* Increment time */
            t_end = t_start + c_lin_inverse(M, S, x_start, x_end);
            t_end = t_end < t_final && fabs(funval) > QUASOARE_EPS ? t_end : t_final;
        }

        /* Increment fluxes during the last interval */
        err_flux = c_lin_fluxes_increment(timestep, gamma, Qstars,
                                          mj_vect, sj_vect,
                                          M, S, t_start, t_end, x_start, x_end,
                                          fluxes);
        if(err_flux>0)
            return err_flux;

        /* Loop for next band */
        funval_prev = M + S * x_end;

        if(isnull(funval_prev)) {
            /* .. f is null => we have reached steady state => complete */
            t_end = t_final;
            jalpha_next = jalpha;
        }
        else {
            t_start = t_end;
            x_start = x_end;
            jalpha = jalpha_next;
        }
    }

    /* Store results */
    *x1 = x_end;
    *niter = nit;

    /* Convergence problem */
    if(notequal(t_final, t_end, QUASOARE_ATOL, QUASOARE_RTOL) && fabs(funval)>QUASOARE_EPS)
        return QUASOARE_NO_CONVERGENCE;

    return 0;
}

int c_lin_reservoir(int nalphas, int nval, int errors,
                    double timestep, double gamma,
                    double x0, double xmin, double xmax,
                    double * alphas,
                    double * Qstars,
                    double * coefs_linear,
                    int * niter, double * x1, double * fluxes) {

    int ierr, t, j;
    int ninputs = 2;
    int nfluxes = 6;
    double store = x0;

    for(t = 0; t < nval; t++) {
        /* Integrate ode */
        ierr = c_lin_integrate(nalphas, alphas, timestep,
                               gamma, &(Qstars[ninputs * t]),
                               coefs_linear,
                               0., store,
                               &(niter[t]),
                               &(x1[t]),
                               &(fluxes[nfluxes * t]));

        /* Manage errors */
        if(ierr > 0) {
            if(errors == 1) {
                return ierr;
            }
            else if (errors == 0 || errors == 2){
                niter[t] = -1;

                x1[t] = store;

                for(j = 0; j < nfluxes; j++)
                    fluxes[nfluxes * t + j] = c_get_nan();
            }
        }

        /* Loop initial state */
        store = x1[t];

        /* Impose constraint */
        store = store < xmin ? xmin : store > xmax ? xmax : store;
        x1[t] = store;
    }

    return 0;
}

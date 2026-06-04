#include "c_quasoare_grad.h"

/* Solution of dS/dt = f*(s) */

double c_grad_forward(double M, double S, double t0, double x0, double t){
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
double c_grad_inverse(double M, double S, double x0, double x1){
    if(isnull(S)){
        return (x1 - x0) / M;
    }
    else {
        return 1./S * log(fabs((S * x1 + M) / (S * x0 + M)));
    }
}


double c_grad_get_qstar(double Qstars[2], double gamma, double timestep, int i) {
    return i < 2 ? Qstars[i] : -gamma / timestep;
}


/* Increment fluxes by integrating f*(x) */
int c_grad_fluxes_increment(double timestep,
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
    double qi;

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

        qi = c_grad_get_qstar(Qstars, gamma, timestep, i) / timestep;
        dflux = qi * (sij * integS + mij * tau);
        fluxes[i] += dflux;

        /* Careful here,
         * Flux bypass is negative because it's leaving the reservoir
         * Demand shortfall is positive if potential demand is negative
         * because it's demand supposed to enter the reservoir. For a
         * downstream reservoir, its sign has to be inverse.
         */
        if(i <= 1)
            fluxes[i + 3] += dflux - qi * tau;
    }

    /* Total outflow = bypasss + outflow */
    fluxes[5] = fluxes[2] + fluxes[3];

    return 0;
}


/* Integrate reservoir equation over 1 time step and compute associated fluxes */
int c_grad_integrate(int nalphas, int nvariables,
                     double * alphas,
                     double timestep,
                     double Qstars[2],
                     double gamma,
                     double * coefs_linear,
                     double t0,
                     double x0,
                     int current_reservoir_id;
                     int *variables_reservoir_id;
                     int *variables_type_id;
                     int *niter,
                     double * x1,
                     double fluxes[6]
                     double *xi0,
                     double *xi1) {

    int idx, i, nit=0, jalpha_next=0, err_flux;
    int nfluxes_primary = 3;
    int nfluxes = 6;
    int ncoefs = 2;

    double M = 0., S = 0.;
    double mij = 0., sij = 0.;
    double funval = 0.;
    double funval_prev = 0.;
    double alpha0, alpha1;
    double alpha_min = alphas[0];
    double alpha_max = alphas[nalphas-1];

    double mj_vect[3];
    double sj_vect[3];

    /* Variables related to gradients */
    int reservoir_id;
    int type_id;
    double w, R, dR, dS, delta, xi_nu;

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

    double qis[3] = {
        c_grad_get_qstar(Qstars, gamma, timestep, 0) / gamma,
        c_grad_get_qstar(Qstars, gamma, timestep, 1) / gamma,
        c_grad_get_qstar(Qstars, gamma, timestep, 2) / gamma
    };

    for(i = 0; i < nfluxes; i++)
        fluxes[i] = 0.;

    for(i = 0; i < nreservoirs * nvariables; i++)
        xi1[i] = xi0[i];

    /* Check data */
    if(Qstars[0] < 0)
        return QUASOARE_NEGATIVE_POTENTIAL_INFLOW_FLUX;

    if(gamma < 0)
        return QUASOARE_NEGATIVE_CAPACITY;

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
        alpha1 = extrapolating_high ? c_get_inf() : alphas[jalpha + 1];

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
            M += mij * qis[i];
            S += sij * qis[i];
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
            x_end = c_grad_forward(M, S, t_start, x_start, t_final);

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
            t_end = t_start + c_grad_inverse(M, S, x_start, x_end);
            t_end = t_end < t_final && fabs(funval) > QUASOARE_EPS ? t_end : t_final;
        }

        /* Increment fluxes during the last interval */
        err_flux = c_grad_fluxes_increment(timestep, gamma, Qstars,
                                          mj_vect, sj_vect,
                                          M, S, t_start, t_end, x_start, x_end,
                                          fluxes);
        if(err_flux>0)
            return err_flux;

        /* Computation of sensitivities */
        delta = t_end - t_start;
        w = exp(S * delta);
        R = M / S;
        for(i = 0; i < nvariables; i++) {
            reservoir_id = variables_reservoir_id[i];
            type_id = variables_type_id[i];

            /* variable belongs to another reservoir */
            if(current_reservoir_id != reservoir_id) {
                dR = 0;
                dS = 0;
            }

            xi1[i] = w * (xi1[i] + dR * (1 - 1. / w) + (x_start + R) * delta * dS);
        }

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

int c_grad_reservoir(int nalphas, int nval, int errors,
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
        ierr = c_grad_integrate(nalphas, alphas, timestep,
                               &(Qstars[ninputs * t]),
                               gamma,
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


int c_grad_model(int nreservoirs, int nalphas, int nval,
                double timestep, double x0,
                double * Qstars,
                double * Qstars_mapping,
                double * alphas,
                double * gammas,
                double * coefs_linear,
                double * x1,
                double * fluxes) {

    int ierr, t, i, j, l, q, p, ka, kq, kc, kx, kf, ko, kw;

    int nfluxes_output = 6;
    int nfluxes_input = 2;
    int nfluxes_function = 3;
    int ncoefs = 2;

    /* we do not store the number of iterations
     * This is a dummy variable */
    int niter[1];

    double store;
    double gamma;
    double w;
    double Qst[2];

    /* timestep loop */
    for(t = 0; t < nval; t++) {

        /* loop through reservoirs */
        for(i = 0; i < nreservoirs; i++) {

            /* Get reservoir capacity */
            gamma = gammas[i];

            /* Data indexes */
            kq = (nreservoirs * t + i) * nfluxes_input;
            kc = nfluxes_function * (nalphas - 1) * ncoefs * i;
            kx = nreservoirs * t + i;
            kf = (nreservoirs * t + i) * nfluxes_output;
            ka = nalphas * i;

            /* Initial condition */
            if(t == 0)
                store = x0;
            else
                store = x1[kx - nreservoirs];

            /* Input data */
            Qst[0] = Qstars[kq];
            Qst[1] = Qstars[kq + 1];

            /* potential flux mapping if it is nan.
             * Otherwise, we consider it as input data.
             * q = 0 -> qstar1 (inflow)
             * q = 1 -> qstar2 (demand)
             * */
            for(q = 0; q < nfluxes_input; q++) {

                if(isnan(Qst[q])) {
                    /* Don't forget to intialise input !*/
                    Qst[q] = 0;

                    for(l = 0; l < nreservoirs; l++) {
                        /* No need to check mapping for itself */
                        if(l == i)
                            continue;

                        for(p = 0; p < nfluxes_output; p++) {

                            /* Check mapping */
                            //kw = nreservoirs * nfluxes_output * nfluxes_input * i
                            //     + nreservoirs * nfluxes_output * q
                            //     + nfluxes_output * l + p;
                            kw = p + nfluxes_output * (l + nreservoirs * (q + nfluxes_input * i));
                            w = Qstars_mapping[kw];
                            if(!isnan(w)) {
                                /* Here, upstream reservoir is contributing
                                 * hence we get the output and map it to the
                                 * current reservoir input.
                                 * */
                                ko = (nreservoirs * t + l) * nfluxes_output + p;
                                Qst[q] += w * fluxes[ko];
                            }

                        }

                    }
                }
            }

            /* Integrate ode */
            ierr = c_grad_integrate(nalphas,
                                   &(alphas[ka]),
                                   timestep,
                                   Qst,
                                   gamma,
                                   &(coefs_linear[kc]),
                                   0.,
                                   store,
                                   niter,
                                   &(x1[kx]),
                                   &(fluxes[kf]));

            /* Basic error management */
            if(ierr > 0)
                return ierr;
        }


    }

    return 0;
}

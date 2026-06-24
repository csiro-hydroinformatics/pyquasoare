#include "c_quasoare_grad.h"

/* Solution of dS/dt = f*(s) */

double wfun(double delta, double S) {
    double x = delta * S;
    if(x > 460.51)
        return 1e200;

    if(x < -460.51)
        return 1e-200;

    if(fabs(x) < 1e-4)
        return 1 + 2 * x / (2 - x + x * x / 6);

    return exp(x);
}

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
            : -M / S +(x0 + M / S) * wfun(tau, S);
    }
    return x1;
}


/* Primitive of 1/f*(s) */
double c_grad_inverse(double M, double S, double x0, double x1){
    if(isnull(S)){
        return (x1 - x0) / M;
    }
    else {
        return 1. / S * log(fabs((S * x1 + M) / (S * x0 + M)));
    }
}


/*
 * deriv :
 * deriv[0] = dM/dη
 * deriv[1] = dS/dη
 * deriv[2] = dμ1j/dη
 * deriv[3] = dμ2j/dη
 * deriv[4] = dμ3j/dη
 * deriv[5] = dσ1j/dη
 * deriv[6] = dσ2j/dη
 * deriv[7] = dσ3j/dη
 */
int c_grad_coefs_derivative(double timestep, int type_id,
                            double gamma,
                            double * flux_scalings,
                            double * Qstars_extended,
                            double * mj_vect,
                            double * sj_vect,
                            double * deriv) {
    int i, dvar;
    double mij, sij;
    double scaling;

    int nfxi = QUASOARE_GRAD_NFLUXES_INPUTS;
    int ncoefs = QUASOARE_GRAD_NPIECEWISEPARAMS;
    int ifxo = QUASOARE_GRAD_IFLUX_OUTFLOW;

    for(dvar = 0; dvar < QUASOARE_GRAD_NDERIV; dvar++)
        deriv[dvar] = 0;

    /* Extract data from vectors */
    if(type_id == QUASOARE_GRAD_VARTYPE_CAPACITY) {
        /* Gradient against capacity of the reservoir */
        deriv[0] = -(Qstars_extended[0] * mj_vect[0]
                     + Qstars_extended[1] * mj_vect[1]) / gamma / gamma;

        deriv[1] = -(Qstars_extended[0] * sj_vect[0]
                     + Qstars_extended[1] * sj_vect[1]) / gamma / gamma;

        /* other derivatives are null because coefficients are
         * independent of Γ */
    }
    else {
        if(type_id < QUASOARE_GRAD_VARTYPE_SCALING_INFLOW
                || type_id > QUASOARE_GRAD_VARTYPE_SCALING_OUTFLOW)
            return QUASOARE_GRAD_TYPEID_NOT_FOUND;

        /* Flux number is assumed to be type of variable - 1 */
        i = type_id - 1;

        mij = mj_vect[i];
        sij = sj_vect[i];
        scaling = flux_scalings[i];

        if(isnull(scaling))
            return QUASOARE_GRAD_ZERO_FLUX_SCALING;

        /* Gradient against scaling applied to inflow, demand o or demand flux function */
        deriv[0] = Qstars_extended[i] * mij / scaling / gamma;
        deriv[1] = Qstars_extended[i] * sij / scaling / gamma;

        deriv[ncoefs + i] = mij / scaling;
        deriv[ncoefs + nfxi + i] = sij / scaling;
    }
    return 0;
}

/* Integrate reservoir equation over 1 time step and compute associated fluxes
 * flux_scalings are scaling factors applied to
 * s[0] : f1 (inputs)
 * s[1] : f2 (demand)
 * s[2] : f3 (outputs)
 * The id of the variables is also matching this definition
 * id = 0 : derivative against reservoir
 * id = 1 : derivative of scaling applied to input flux function
 * id = 2 : derivative of scaling applied to demand flux function
 * id = 3 : derivative of scaling applied to output flux function
 *
 * input_mapping : vector of nreservoirs × 6 output fluxes × nreservoirs × 2 input fluxes
 *   columns = [res1 / w(out1) w(out2) .. w(out6)] [res2 / w(out1) ..] ...
 *   rows = [res1 / (inp 1) (inp2)] [res2 / (inp1) (inp2)]
 *
 * dfluxes : vector of nvariables × 6 x nreservoirs
 * */
int c_grad_integrate(int nalphas, int nreservoirs, int nvariables,
                     double * alphas,
                     double timestep,
                     double gamma,
                     double *input_mapping,
                     double * Qstars,
                     double * flux_scalings,
                     double * coefs_linear,
                     double t0,
                     double x0,
                     double *dx0,
                     int current_reservoir_id,
                     int *variables_reservoir_id,
                     int *variables_type_id,
                     int *niter,
                     double * x1,
                     double * dx1,
                     double * fluxes,
                     double * dfluxes) {
    int ierr;
    int i, dvar, res, out, inp, idx_c, idx_m, idx_m0;
    int idx_df, idx_df0;
    int nit=0, jalpha_next=0;

    int nfxi = QUASOARE_GRAD_NFLUXES_INPUTS;
    int nfxp = QUASOARE_GRAD_NFLUXES_POTENTIAL;
    int nfxo = QUASOARE_GRAD_NFLUXES_OUTPUTS;
    int ncoefs = QUASOARE_GRAD_NPIECEWISEPARAMS;

    int ifxi = QUASOARE_GRAD_IFLUX_INFLOW;
    int ifxd = QUASOARE_GRAD_IFLUX_DEMAND;
    int ifxb = QUASOARE_GRAD_IFLUX_BYPASS;
    int ifxs = QUASOARE_GRAD_IFLUX_SHORTFALL;
    int ifxo = QUASOARE_GRAD_IFLUX_OUTFLOW;
    int ifxt = QUASOARE_GRAD_IFLUX_TOTALOUT;

    double M = 0., S = 0.;
    double mij = 0., sij = 0.;
    double flx;
    double funval = 0.;
    double funval_prev = 0.;
    double alpha0, alpha1;
    double alpha_min = alphas[0];
    double alpha_max = alphas[nalphas-1];

    double mj_vect[QUASOARE_GRAD_NFLUXES_INPUTS];
    double sj_vect[QUASOARE_GRAD_NFLUXES_INPUTS];

    double input_mapping_weight;

    /* Extended Qstars */
    double Qstars_extended[QUASOARE_GRAD_NFLUXES_INPUTS];
    Qstars_extended[0] = Qstars[0];
    Qstars_extended[1] = Qstars[1];
    Qstars_extended[2] = -gamma / timestep;
    double qi;

    /* Variables related to gradients */
    int reservoir_id, type_id;
    double w, R, dR, dS, dM, tau;
    double dx_start, dx_end;
    double dQstars_extended[QUASOARE_GRAD_NFLUXES_INPUTS];
    double integ_x;
    double integ_fx[QUASOARE_GRAD_NFLUXES_INPUTS];
    double integ_dfx[QUASOARE_GRAD_NFLUXES_INPUTS];
    /*... dM dS dmij[i=1,2,3] dsig[i=1,2,3] : total 8 doubles */
    double deriv_coefs[QUASOARE_GRAD_NDERIV];
    double dmuij[QUASOARE_GRAD_NFLUXES_INPUTS];
    double dsij[QUASOARE_GRAD_NFLUXES_INPUTS];
    double H, G;

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

    for(i = 0; i < nfxo; i++)
        fluxes[i] = 0.;

    for(i = 0; i < nvariables; i++)
        dx1[i] = dx0[i];

    /* Check data */
    if(Qstars[0] < 0)
        return QUASOARE_NEGATIVE_POTENTIAL_INFLOW_FLUX;

    if(gamma <= 0)
        return QUASOARE_CAPACITY_IS_NEGATIVE_OR_NULL;

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
        for(i = 0 ; i < nfxi; i++) {
            /* Compute flux coefficients */

            if(extrapolating_low)
                idx_c = (nalphas - 1) * ncoefs * i;
            else if(extrapolating_high)
                idx_c = (nalphas - 1) * ncoefs * i + ncoefs * (nalphas - 2);
            else
                idx_c = (nalphas - 1) * ncoefs * i + ncoefs * jalpha;

            sij = coefs_linear[idx_c];        // Slope
            mij = coefs_linear[idx_c + 1];    // Intercept

            /* All flux functions are assumed positive */
            if(1 - extrapolating &&
                    (mij + sij * alpha0 < 0 || mij + sij * alpha1 < 0))
                return QUASOARE_NEGATIVE_FLUX_FUNCTION;

            /* apply scaling */
            mij *= flux_scalings[i];
            sij *= flux_scalings[i];

            /* Store flux coefficents */
            mj_vect[i] = mij;
            sj_vect[i] = sij;

            /* Add coefficients to obtain approximation of reservoir equation
             * function */
            qi = Qstars_extended[i] / gamma;
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

        if(!isfinite(x_end))
            return QUASOARE_STORE_LEVEL_INFINITE;

        /* Time increment variable */
        tau = t_end - t_start;

        /* Compute the integral of x(t) over the time interval */
        if(notnull(S))
            integ_x = (x_end - x_start - M * tau) / S;
        else
            integ_x = (x_start + M * tau / 2) * tau;

        /* Increment fluxes during the last interval */
        for(i = 0; i < nfxi; i++){
            integ_fx[i] = sj_vect[i] * integ_x + mj_vect[i] * tau;
            flx = Qstars_extended[i] / timestep * integ_fx[i];
            fluxes[i] += flx;

            /* Careful here,
             * Flux bypass is negative because it's leaving the reservoir
             * Demand shortfall is positive if potential demand is negative
             * because the demand flux is negative (it leaves the reservoir),
             * hence the shortfall is positive (it corresponds to a flux
             * entering the reservoir).
             * */
            if(i == ifxi || i == ifxd)
                fluxes[i + nfxi] += flx - Qstars_extended[i] / timestep * tau;
        }

        /* Total outflow = bypasss + outflow */
        fluxes[ifxt] = fluxes[ifxb] + fluxes[ifxo];

        /******* Computation of sensitivities *******/
        w = wfun(tau, S);
        R = M / S;
        for(dvar = 0; dvar < nvariables; dvar++) {
            reservoir_id = variables_reservoir_id[dvar];
            type_id = variables_type_id[dvar];

            /* Baseline indexes for flux mappings */
            idx_m0 = nfxp * nreservoirs * nfxo * reservoir_id;

            /* Baseline indexes for flux derivatives */
            idx_df0 = nfxo * nreservoirs * dvar;

            if (current_reservoir_id != reservoir_id) {
                /* -- variable belongs to a different reservoir -- */

                /* 1. compute dQstars_extendedar[dvar] / dη based on upstream flux derivatives
                 * and their mapping to the current reservoir */
                dQstars_extended[nfxi - 1] = 0;

                for(inp = 0; inp < nfxp; inp++){
                    /* initialise derivative */
                    dQstars_extended[inp] = 0;

                    /* upstream contributions */
                    for(res = 0; res < nreservoirs; res++) {
                        for(out = 0; out < nfxo; out++) {
                            idx_m = idx_m0 + nreservoirs * nfxo * inp +  nfxo* res + out;
                            input_mapping_weight = input_mapping[idx_m];
                            if(notnull(input_mapping_weight)) {
                                idx_df = idx_df0 + nfxo  * res + out;
                                dQstars_extended[inp] += input_mapping_weight * dfluxes[idx_df];
                            }
                        }
                    }
                }

                /* 2. use this result to compute dM and dS */
                dM = (mj_vect[0] * dQstars_extended[0]
                      + mj_vect[1] * dQstars_extended[1]) / gamma;

                dS = (sj_vect[0] * dQstars_extended[0]
                      + sj_vect[1] * dQstars_extended[1]) / gamma;

                /* 3. dμij and dσij is null because η belongs to another
                 * reservoir */
                for(inp = 0; inp < nfxi; inp ++){
                    dmuij[inp] = 0;
                    dsij[inp] = 0;
                }

            } else {
                /* -- variable belongs to the same reservoir --- */

                /* Derivative of Qstars */
                dQstars_extended[0] = 0;
                dQstars_extended[1] = 0;

                if(type_id == QUASOARE_GRAD_VARTYPE_CAPACITY)
                    /* Derivative of Qstars for outflow (i.e. -Γ/Δ)
                     * against reservoir capacity (i.e. -1/Δ) */
                    dQstars_extended[2] = -1. / timestep;
                else
                    dQstars_extended[2] = 0.;

                /* Derivative of M, S, μij and σij vs η */
                ierr = c_grad_coefs_derivative(timestep,
                                               type_id, gamma,
                                               flux_scalings,
                                               Qstars_extended,
                                               mj_vect, sj_vect,
                                               deriv_coefs);
                if(ierr > 0)
                    return ierr;

                dM = deriv_coefs[0];
                dS = deriv_coefs[1];
                for(inp = 0; inp < nfxi; inp ++){
                    dmuij[inp] = deriv_coefs[ncoefs + inp];
                    dsij[inp] = deriv_coefs[ncoefs + nfxi + inp];
                }
            }

            dx_start = dx1[dvar];

            /* --- computation of integral of df(x)/dη --- */
            if(notequal(x_start, x_end, QUASOARE_EPS, 0.)) {
                if(notnull(S)) {
                    /* Compute dR and then dx_end assuming S≠0 */
                    dR = (dM - R * dS) / S;
                    dx_end = w * dx_start + dR * (w - 1.) + (x_start + R) * tau * dS * w;

                    /* compute integral of dfi(x)/dη between tj and tj+1 */
                    for(i = 0; i < nfxi; i++) {
                        integ_dfx[i] = (dmuij[i] - sj_vect[i] * dR - R * dsij[i]) * tau
                            + (dsij[i] - sj_vect[i] / S * dS) * (x_end - x_start)
                            + sj_vect[i] / S * (dx_end - dx_start);
                    }

                } else {
                    /* Compute dx_end assuming S=0 */
                    dx_end = dx_start + dM * tau;

                    /* compute integral of dfi(x)/dη between tj and tj+1 */
                    for(i = 0; i < nfxi; i++) {
                        integ_dfx[i] = dmuij[i] * tau
                            + (M * dsij[i] + sj_vect[i] * dM) * tau * tau / 2;
                    }
                }

            } else {
                dx_end = dx_start;

                for(i = 0; i < nfxi; i++)
                    integ_dfx[i] = 0;
            }

            //fprintf(stdout,
            //        "\ngam=%2.5e Qs0=%2.5e Qs1=%2.5e tau=%2.5e M=%2.5e S=%2.5e\n"
            //        " dM=%2.5e dS=%2.5e dR=%2.5e w=%2.5e\n"
            //        " x0=%2.5e x1=%2.5e (!=x0?%d) dx0=%2.5e dx1=%2.5e\n\n",
            //        gamma, Qstars[0], Qstars[1], tau, M, S, dM, dS, dR,
            //        w, x_start, x_end, notequal(x_start, x_end, QUASOARE_EPS, 0.),
            //        dx_start, dx_end);

            if(!isfinite(dx_end))
                return QUASOARE_GRAD_IS_INFINITE;

            if(isnan(dx_end))
                return QUASOARE_GRAD_IS_NAN;

            /* Compute the two terms G and H related to flux derivatives */
            for(i = 0; i < nfxo ; i++) {
                if(i <= ifxi) {
                    /* Inflow, demand and outflow fluxes */
                    G = dQstars_extended[i] / timestep * integ_fx[i];
                    H = Qstars_extended[i] * integ_dfx[i] / timestep;

                } else if (i == ifxb || i == ifxs) {
                    /* Bypass and demand shortfall */
                    G = dQstars_extended[i - nfxi] * (1. - integ_fx[i - nfxi] / timestep);
                    H = - Qstars_extended[i - nfxi] * integ_dfx[i - nfxi] / timestep;

                } else if (i == ifxt) {
                    /* Total outflow */
                    G = dQstars_extended[ifxo] / timestep * integ_fx[ifxo]
                        + dQstars_extended[ifxi] * (1. - integ_fx[ifxi] / timestep);
                    H = Qstars_extended[ifxo] * integ_dfx[ifxo] / timestep
                        - Qstars_extended[ifxi] * integ_dfx[ifxi] / timestep;
                }

                /* Final storage of flux derivative for the current reservoir */
                idx_df = idx_df0 + nfxo  * current_reservoir_id + i;
                dfluxes[idx_df] = G + H;
            }

            /* Store loop variables */
            dx1[dvar] = dx_end;
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

int c_grad_reservoir(int nalphas, int nval,
                    int errors,
                    double * alphas,
                    double timestep,
                    double gamma,
                    double * flux_scalings,
                    double x0,
                    double * dx0,
                    double * Qstars,
                    double * coefs_linear,
                    int * niter,
                    double * x1,
                    double * dx1,
                    double * fluxes,
                    double * dfluxes) {
    int ierr, t, dvar, i, im, idx_dx, idx_f, idx_df;
    double store = x0;
    double * dx_ini = dx0;

    int nfxp = QUASOARE_GRAD_NFLUXES_POTENTIAL;
    int nfxo = QUASOARE_GRAD_NFLUXES_OUTPUTS;

    /* configure model for a single reservoir */
    int current_reservoir_id = 0;

    /* .. we assume that we compute the derivative
     * agains all possible variables types */
    int nvariables = QUASOARE_GRAD_VARTYPE_NUMBER;
    int variables_reservoir_id[QUASOARE_GRAD_VARTYPE_NUMBER];
    int variables_type_id[QUASOARE_GRAD_VARTYPE_NUMBER];

    for(dvar = 0; dvar < nvariables; dvar++){
        variables_reservoir_id[dvar] = 0;
        variables_type_id[dvar] = dvar;
    }

    /* .. only one reservoir with no input mapping from others */
    int nreservoirs = 1;
    double input_mapping[QUASOARE_GRAD_NFLUXES_POTENTIAL * QUASOARE_GRAD_NFLUXES_OUTPUTS];
    for(im = 0; im < QUASOARE_GRAD_NFLUXES_POTENTIAL * QUASOARE_GRAD_NFLUXES_OUTPUTS; im++)
        input_mapping[im] = 0.;

    /* Time loop */
    for(t = 0; t < nval; t++) {
        /* indexes */
        idx_dx = nvariables * t;
        idx_f = nreservoirs * nfxo * t;
        idx_df = nvariables * idx_f; // nvar × nres × nfx_out × t

        /* Integrate ode */
        ierr = c_grad_integrate(nalphas, nreservoirs, nvariables,
                               alphas, timestep,
                               gamma,
                               input_mapping,
                               &(Qstars[nfxp * t]),
                               flux_scalings,
                               coefs_linear,
                               0., store,
                               dx_ini,
                               current_reservoir_id,
                               variables_reservoir_id,
                               variables_type_id,
                               &(niter[t]),
                               &(x1[t]),
                               &(dx1[idx_dx]),
                               &(fluxes[idx_f]),
                               &(dfluxes[idx_df]));

        /* Manage errors */
        if(ierr > 0) {
            if(errors == 1)
                return ierr;

            else if (errors == 0 || errors == 2){
                niter[t] = -1;

                x1[t] = store;

                for(i = 0; i < nfxo; i++)
                    fluxes[idx_f + i] = c_get_nan();

                for(dvar = 0; dvar < nvariables; dvar++)
                    dx1[idx_dx + dvar] = c_get_nan();
            }
        }

        /* Loop initial states */
        store = x1[t];
        dx_ini = &(dx1[idx_dx]);
    }

    return 0;
}


int c_grad_model(int nalphas, int nreservoirs, int nvariables, int nval,
                 int errors,
                 double * alphas,
                 double timestep,
                 double * gammas,
                 double * input_mapping,
                 double * flux_scalings,
                 double x0,
                 double * dx0,
                 int *variables_reservoir_id,
                 int *variables_type_id,
                 double * Qstars,
                 double * coefs_linear,
                 double * x1,
                 double * dx1,
                 double * fluxes,
                 double * dfluxes) {

    int ierr, t, dvar;
    int idx_q, idx_c, idx_x, idx_f, idx_a, idx_dx, idx_dxp;
    int idx_m, idx_m0, idx_f0, idx_df, idx_fs;
    int inp, out, res, resu, nres_times_t;

    int nfxi = QUASOARE_GRAD_NFLUXES_INPUTS;
    int nfxp = QUASOARE_GRAD_NFLUXES_POTENTIAL;
    int nfxo = QUASOARE_GRAD_NFLUXES_OUTPUTS;
    int ncoefs = QUASOARE_GRAD_NPIECEWISEPARAMS;

    int current_reservoir_id;

    /* we do not store the number of iterations
     * This is a dummy variable */
    int niter[1];

    double store;
    double gamma;
    double w;
    double Qst[QUASOARE_GRAD_NFLUXES_POTENTIAL];

    /* timestep loop */
    for(t = 0; t < nval; t++) {

        /* loop through reservoirs */
        for(res = 0; res < nreservoirs; res++) {
            current_reservoir_id = res;

            /* Get reservoir capacity */
            gamma = gammas[res];

            /* Data indexes */
            nres_times_t = nreservoirs * t;
            idx_q = (nres_times_t + res) * nfxp;
            idx_fs = res * nfxo;
            idx_c = nfxi * (nalphas - 1) * ncoefs * res;
            idx_x = nres_times_t + res;
            idx_a = nalphas * res;
            idx_dx = (nres_times_t + res) * nvariables;
            idx_dxp = (nres_times_t - nreservoirs + res) * nvariables;
            idx_f = (nres_times_t + res) * nfxo;
            idx_df = (nres_times_t + res) * nfxo * nvariables;

            /* Initial condition */
            if(t == 0)
                store = x0;
            else {
                store = x1[idx_x - nreservoirs];
                dx0 = &(dx1[idx_dxp]);
            }

            /* Input data */
            Qst[0] = Qstars[idx_q];
            Qst[1] = Qstars[idx_q + 1];

            /* Baseline flux mapping index */
            idx_m0 = nfxp * nreservoirs * nfxo * res;

            /* ... applied potential flux mapping if the above data is nan.
             * Otherwise, we consider it as input data.
             * */
            for(inp = 0; inp < nfxp; inp++) {
                if(isnan(Qst[inp])) {
                    /* Don't forget to intialise input !*/
                    Qst[inp] = 0;

                    for(resu = 0; resu < nreservoirs; resu++) {
                        /* No need to checdvar mapping for itself */
                        if(resu == res)
                            continue;

                        idx_f0 = (nreservoirs * t + resu) * nfxo;
                        for(out = 0; out < nfxo; out++) {
                            /* Check mapping is not null */
                            idx_m = idx_m0 + nreservoirs * nfxo * inp +  nfxo* res + out;
                            w = input_mapping[idx_m];
                            if(notnull(w)) {
                                /* Here, upstream reservoir is contributing
                                 * hence we get the output and map it to the
                                 * current reservoir input.
                                 * */
                                Qst[inp] += w * fluxes[idx_f0 + out];
                            }

                        }

                    }
                }
            }

            /* Integrate ode */
            ierr = c_grad_integrate(nalphas, nreservoirs, nvariables,
                                    &(alphas[idx_a]),
                                    timestep,
                                    gamma,
                                    input_mapping,
                                    Qst,
                                    &(flux_scalings[idx_fs]),
                                    &(coefs_linear[idx_c]),
                                    0.,
                                    store,
                                    dx0,
                                    current_reservoir_id,
                                    variables_reservoir_id,
                                    variables_type_id,
                                    niter,
                                    &(x1[idx_x]),
                                    &(dx1[idx_dx]),
                                    &(fluxes[idx_f]),
                                    &(dfluxes[idx_df]));

            /* error management */
            if(ierr > 0) {
                if(errors == 1)
                    return ierr;

                else if (errors == 0 || errors == 2){
                    niter[t] = -1;
                    x1[t] = store;

                    for(out = 0; out < nfxo; out++)
                        fluxes[nfxo * t + out] = c_get_nan();

                    for(dvar = 0; dvar < nvariables; dvar++)
                        dx1[nvariables * t + dvar] = c_get_nan();
                }
            }
        }
    }

    return 0;
}

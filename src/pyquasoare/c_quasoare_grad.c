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
                            double * Qstars_ext,
                            double * mj_vect,
                            double * sj_vect,
                            double * deriv) {
    int i, dvar;
    double mij, sij;
    double scaling;

    int nfxi = QUASOARE_GRAD_NFLUXES_INPUTS;
    int ncoefs = QUASOARE_GRAD_NPIECEWISEPARAMS;
    int ifxdr = QUASOARE_GRAD_IFLUX_DRAINAGE;

    for(dvar = 0; dvar < QUASOARE_GRAD_NDERIV; dvar++)
        deriv[dvar] = 0;

    /* Extract data from vectors */
    if(type_id == QUASOARE_GRAD_VARTYPE_CAPACITY) {
        /* Gradient against capacity of the reservoir */
        deriv[0] = -(Qstars_ext[0] * mj_vect[0]
                     + Qstars_ext[1] * mj_vect[1]) / gamma / gamma;

        deriv[1] = -(Qstars_ext[0] * sj_vect[0]
                     + Qstars_ext[1] * sj_vect[1]) / gamma / gamma;

        /* other derivatives are null because coefficients are
         * independent of Γ */
    }
    else {
        if(type_id < QUASOARE_GRAD_VARTYPE_SCALING_INFLOW
                || type_id > QUASOARE_GRAD_VARTYPE_SCALING_DRAINAGE)
            return QUASOARE_GRAD_TYPEID_NOT_FOUND;

        /* Flux number is assumed to be type of variable - 1 */
        i = type_id - 1;

        mij = mj_vect[i];
        sij = sj_vect[i];
        scaling = flux_scalings[i];

        if(isnull(scaling))
            return QUASOARE_GRAD_ZERO_FLUX_SCALING;

        /* Gradient against scaling applied to inflow, demand o or demand flux function */
        deriv[0] = Qstars_ext[i] * mij / scaling / gamma;
        deriv[1] = Qstars_ext[i] * sij / scaling / gamma;

        deriv[ncoefs + i] = mij / scaling;
        deriv[ncoefs + nfxi + i] = sij / scaling;
    }
    return 0;
}


int c_grad_qstars_derivative(int nreservoirs, int nvariables,
                             int * input_mapping,
                             double timestep,
                             int current_reservoir_id,
                             int *variables_reservoir_id,
                             int *variables_type_id,
                             double * dfluxes,
                             double * dQstars_ext) {
    int dvar, idx_m0, idx_m, idx_df, idx_df0, idx_dq, ulink, inp;
    int reservoir_id, type_id, resup_id, resup_ofx_id, resdown_pfx_id;
    double input_mapping_weight;
    double input_mapping_scaling_factor = (double)(QUASOARE_GRAD_INPUT_MAPPING_SCALINGFACTOR);

    int nfxi = QUASOARE_GRAD_NFLUXES_INPUTS;
    int nfxo = QUASOARE_GRAD_NFLUXES_OUTPUTS;
    int nipmd = QUASOARE_GRAD_INPUT_MAPPING_LASTDIM;
    int ifxdr = QUASOARE_GRAD_IFLUX_DRAINAGE;

    /* Baseline index for input mapping corresponding
     * to current reservoir */
    idx_m0 = nreservoirs * nfxo * nipmd * current_reservoir_id;

    /* Loop on variables */
    for(dvar = 0; dvar < nvariables; dvar++) {
        reservoir_id = variables_reservoir_id[dvar];
        type_id = variables_type_id[dvar];
        idx_dq = nfxi * dvar;

        /* Baseline index for flux derivatives */
        idx_df0 = nfxo * nreservoirs * dvar;

        /* initialise derivatives */
        for(inp = 0; inp < nfxi; inp++)
            dQstars_ext[idx_dq + inp] = 0;

        if(reservoir_id != current_reservoir_id) {
            /* Variable does not belong to the current reservoir.
             * Compute dQstars_timestep_extar[dvar] / dη based
             * on upstream flux derivatives and their mapping to
             * the current reservoir */

             /* Derivative of upstream contributions.
              * This loops over the maximum number of connections
              * but it stops as soon as the upstream reservoir id is
              * -1. */
             for(ulink = 0; ulink < nreservoirs * nfxo; ulink++) {
                 idx_m = idx_m0 + ulink * nipmd;
                 resup_id = input_mapping[idx_m];

                 /* No more links ... we break */
                 if(resup_id < 0)
                     break;

                 resup_ofx_id = input_mapping[idx_m + 1];
                 resdown_pfx_id = input_mapping[idx_m + 2];

                 /* here we scale the weight using the scaling factor */
                 input_mapping_weight = (double)(input_mapping[idx_m + 3])
                     / input_mapping_scaling_factor;

                 idx_df = idx_df0 + nfxo  * resup_id + resup_ofx_id;
                 dQstars_ext[idx_dq + resdown_pfx_id]
                     += input_mapping_weight * dfluxes[idx_df];
            }

        } else {
            /* Variable belongs to the current reservoir.
             * Only non zero derivative is derivative against capacity Γ */
            if(type_id == QUASOARE_GRAD_VARTYPE_CAPACITY)
                dQstars_ext[idx_dq + ifxdr] = -1. / timestep;
        }
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
                     int *input_mapping,
                     double * Qstars,
                     double * dQstars_ext,
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
    int inp, out, dvar, idx_c, idx_m, idx_m0, idx_dq;
    int idx_df, idx_df0;
    int nit=0, jalpha_next=0;

    int nfxi = QUASOARE_GRAD_NFLUXES_INPUTS;
    int nfxp = QUASOARE_GRAD_NFLUXES_POTENTIAL;
    int nfxo = QUASOARE_GRAD_NFLUXES_OUTPUTS;
    int nipmd = QUASOARE_GRAD_INPUT_MAPPING_LASTDIM;
    int ncoefs = QUASOARE_GRAD_NPIECEWISEPARAMS;

    int ifxi = QUASOARE_GRAD_IFLUX_INFLOW;
    int ifxd = QUASOARE_GRAD_IFLUX_DEMAND;
    int ifxdr = QUASOARE_GRAD_IFLUX_DRAINAGE;
    int ifxb = QUASOARE_GRAD_IFLUX_BYPASS;
    int ifxs = QUASOARE_GRAD_IFLUX_SHORTFALL;
    int ifxo = QUASOARE_GRAD_IFLUX_OUTFLOW;
    int ifx_original;

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

    /* Extended Qstars_timestep */
    double Qstars_ext[QUASOARE_GRAD_NFLUXES_INPUTS];
    Qstars_ext[0] = Qstars[0];
    Qstars_ext[1] = Qstars[1];
    Qstars_ext[2] = -gamma / timestep;
    double qi;

    /* Variables related to gradients */
    int reservoir_id, type_id;
    double w, R, dR, dS, dM, tau;
    double dx_start, dx_end;
    double integ_x, integ_dx;
    double integ_fx[QUASOARE_GRAD_NFLUXES_INPUTS];
    double integ_dfx[QUASOARE_GRAD_NFLUXES_INPUTS];
    /*... dM dS dmij[i=1,2,3] dsig[i=1,2,3] : total 8 doubles */
    double deriv_coefs[QUASOARE_GRAD_NDERIV];
    double dmuij[QUASOARE_GRAD_NFLUXES_INPUTS];
    double dsij[QUASOARE_GRAD_NFLUXES_INPUTS];
    double H, G, dQs;

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

    /* Check data */
    if(Qstars[0] < 0)
        return QUASOARE_NEGATIVE_POTENTIAL_INFLOW_FLUX;

    if(gamma <= 0)
        return QUASOARE_CAPACITY_IS_NEGATIVE_OR_NULL;

    /* Compute Qstars derivative */
    c_grad_qstars_derivative(nreservoirs, nvariables,
                             input_mapping, timestep,
                             current_reservoir_id,
                             variables_reservoir_id,
                             variables_type_id,
                             dfluxes,
                             dQstars_ext);

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
        for(inp = 0 ; inp < nfxi; inp++) {
            /* Compute flux coefficients */
            if(extrapolating_low)
                idx_c = (nalphas - 1) * ncoefs * inp;
            else if(extrapolating_high)
                idx_c = (nalphas - 1) * ncoefs * inp + ncoefs * (nalphas - 2);
            else
                idx_c = (nalphas - 1) * ncoefs * inp + ncoefs * jalpha;

            sij = coefs_linear[idx_c];        // Slope
            mij = coefs_linear[idx_c + 1];    // Intercept

            /* All flux functions are assumed positive */
            if(1 - extrapolating &&
                    (mij + sij * alpha0 < 0 || mij + sij * alpha1 < 0))
                return QUASOARE_NEGATIVE_FLUX_FUNCTION;

            /* apply scaling */
            mij *= flux_scalings[inp];
            sij *= flux_scalings[inp];

            /* Store flux coefficents */
            mj_vect[inp] = mij;
            sj_vect[inp] = sij;

            /* Add coefficients to obtain approximation of reservoir equation
             * function */
            qi = Qstars_ext[inp] / gamma;
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
        for(inp = 0; inp < nfxi; inp++){
            integ_fx[inp] = sj_vect[inp] * integ_x + mj_vect[inp] * tau;
            flx = Qstars_ext[inp] / timestep * integ_fx[inp];

            if(nit == 1)
                fluxes[inp] = flx;
            else
                fluxes[inp] += flx;

            /* Careful here,
             * Flux bypass is negative because it's leaving the reservoir
             * Demand shortfall is positive if potential demand is negative
             * because the demand flux is negative (it leaves the reservoir),
             * hence the shortfall is positive (it corresponds to a flux
             * entering the reservoir).
             * */
            if(inp == ifxi || inp == ifxd)
                fluxes[inp + nfxi] += flx - Qstars_ext[inp] / timestep * tau;
        }

        /* Total outflow = bypasss + drainage */
        fluxes[ifxo] = fluxes[ifxb] + fluxes[ifxdr];

        /******* Computation of sensitivities *******/

        /* Baseline index for input mapping */
        idx_m0 = nreservoirs * nfxo * nipmd * current_reservoir_id;

        /* Useful constants */
        w = wfun(tau, S);
        R = M / S;

        /* Loop over independent variables */
        for(dvar = 0; dvar < nvariables; dvar++) {
            reservoir_id = variables_reservoir_id[dvar];
            type_id = variables_type_id[dvar];

            /* Baseline index for flux derivatives */
            idx_df0 = nfxo * nreservoirs * dvar;

            if (current_reservoir_id != reservoir_id) {
                /* -- variable belongs to a different reservoir -- */
                dM = (mj_vect[0] * dQstars_ext[0]
                      + mj_vect[1] * dQstars_ext[1]) / gamma;

                dS = (sj_vect[0] * dQstars_ext[0]
                      + sj_vect[1] * dQstars_ext[1]) / gamma;

                /* 3. dμij and dσij is null because η belongs to another
                 * reservoir */
                for(inp = 0; inp < nfxi; inp ++){
                    dmuij[inp] = 0;
                    dsij[inp] = 0;
                }

            } else {
                /* -- variable belongs to the same reservoir --- */

                /* Derivative of M, S, μij and σij vs η */
                ierr = c_grad_coefs_derivative(timestep,
                                               type_id, gamma,
                                               flux_scalings,
                                               Qstars_ext,
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

            /* Initialise dx */
            if(nit == 1)
                dx_start = dx0[dvar];
            else
                dx_start = dx1[dvar];

            /* --- computation of ξ, integral of ξ, and integral of df(x)/dη --- */
            if(notequal(x_start, x_end, QUASOARE_EPS, 0.)) {
                if(notnull(S)) {
                    /* Compute dR and then dx_end assuming S≠0 */
                    dR = (dM - R * dS) / S;
                    dx_end = w * dx_start + dR * (w - 1.) + (x_start + R) * tau * dS * w;
                    integ_dx = (dx_end - dx_start - dM * tau - dS * integ_x) / S;

                } else {
                    /* Compute dx_end assuming S=0 */
                    dx_end = dx_start + dM * tau;
                    integ_dx = dM * tau * tau / 2;
                }

                for(inp = 0; inp < nfxi; inp++)
                    integ_dfx[inp] = dmuij[inp] * tau + dsij[inp] * integ_x + sj_vect[inp] * integ_dx;

            } else {
                dx_end = dx_start;

                for(inp = 0; inp < nfxi; inp++)
                    integ_dfx[inp] = 0;
            }

            if(!isfinite(dx_end))
                return QUASOARE_GRAD_IS_INFINITE;

            if(isnan(dx_end))
                return QUASOARE_GRAD_IS_NAN;

            /* Baseline index for Qstars derivatives */
            idx_dq = nfxi * dvar;

            /* Compute the two terms G and H related to flux derivatives */
            for(out = 0; out < nfxo ; out++) {
                if(out < nfxi) {
                    /* Inflow, demand and drainage fluxes */
                    G = dQstars_ext[idx_dq + out] / timestep * integ_fx[out];
                    H = Qstars_ext[out] * integ_dfx[out] / timestep;

                } else if (out == ifxb || out == ifxs) {
                    /* Bypass and demand shortfall */
                    ifx_original = out == ifxb ? ifxi : ifxd;
                    G = dQstars_ext[idx_dq + ifx_original] * (integ_fx[ifx_original] - tau) / timestep;
                    H = Qstars_ext[ifx_original] * integ_dfx[ifx_original] / timestep;

                } else if (out == ifxo) {
                    /* Total outflow  = drainage + bypass*/
                    G = (dQstars_ext[idx_dq + ifxdr] * integ_fx[ifxdr]
                        + dQstars_ext[idx_dq + ifxi] * (integ_fx[ifxi] - tau))/ timestep;
                    H = (Qstars_ext[ifxdr] * integ_dfx[ifxdr] + Qstars_ext[ifxi] * integ_dfx[ifxi]) / timestep;
                }

                /* Final storage of flux derivative for the current reservoir */
                idx_df = idx_df0 + nfxo  * current_reservoir_id + out;

                /* Initialse dflux for the first iteration */
                if(nit == 1)
                    dfluxes[idx_df] = G + H;
                else
                    dfluxes[idx_df] += G + H;
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
                    double * dQstars_ext,
                    double * coefs_linear,
                    int * niter,
                    double * x1,
                    double * dx1,
                    double * fluxes,
                    double * dfluxes) {
    int ierr, t, dvar, i, idx_dx, idx_f, idx_df;
    double store = x0;
    double * dx_ini = dx0;

    int nfxp = QUASOARE_GRAD_NFLUXES_POTENTIAL;
    int nfxo = QUASOARE_GRAD_NFLUXES_OUTPUTS;

    /* configure model for a single reservoir */
    int current_reservoir_id = 0;

    /* .. only one reservoir with no input mapping from others */
    int nreservoirs = 1;
    int input_mapping[QUASOARE_GRAD_INPUT_MAPPING_LASTDIM] = {-1};

    /* .. we assume that we compute the derivative
     * agains all possible variable types */
    int nvariables = QUASOARE_GRAD_VARTYPE_NUMBER;
    int variables_reservoir_id[QUASOARE_GRAD_VARTYPE_NUMBER];
    int variables_type_id[QUASOARE_GRAD_VARTYPE_NUMBER];

    for(dvar = 0; dvar < nvariables; dvar++){
        variables_reservoir_id[dvar] = 0;
        variables_type_id[dvar] = dvar;
    }

    /* Time loop */
    for(t = 0; t < nval; t++) {
        /* indexes */
        idx_dx = nvariables * t;
        idx_f = nfxo * t;
        idx_df = nvariables * idx_f; // nvar × nfx_out × t

        /* Integrate ode */
        ierr = c_grad_integrate(nalphas, nreservoirs, nvariables,
                                alphas, timestep,
                                gamma,
                                input_mapping,
                                &(Qstars[nfxp * t]),
                                dQstars_ext,
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
                 int *input_mapping,
                 double * flux_scalings,
                 double x0,
                 double * dx0,
                 int *variables_reservoir_id,
                 int *variables_type_id,
                 double *Qstars,
                 double *dQstars_ext,
                 double *coefs_linear,
                 double *x1,
                 double *dx1,
                 double *fluxes,
                 double *dfluxes) {

    int ierr, t, dvar;
    int idx_q, idx_c, idx_x, idx_f, idx_a, idx_dx, idx_dxp;
    int idx_m, idx_m0, idx_up_f, idx_df, idx_fs;
    int inp, out, res, resu, nres_times_t;

    int nfxi = QUASOARE_GRAD_NFLUXES_INPUTS;
    int nfxp = QUASOARE_GRAD_NFLUXES_POTENTIAL;
    int nfxo = QUASOARE_GRAD_NFLUXES_OUTPUTS;
    int nipmd = QUASOARE_GRAD_INPUT_MAPPING_LASTDIM;
    int ncoefs = QUASOARE_GRAD_NPIECEWISEPARAMS;

    int current_reservoir_id;

    int ulink, resup_id, resup_ofx_id, resdown_pfx_id;
    double input_mapping_scaling_factor = (double)(QUASOARE_GRAD_INPUT_MAPPING_SCALINGFACTOR);
    double input_mapping_weight;

    /* Variables related to gradients */
    int reservoir_id, type_id;
    double w, R, dR, dS, dM, tau;
    double dx_start, dx_end;
    double dQstars_timestep_ext[QUASOARE_GRAD_NFLUXES_INPUTS];
    double integ_x;
    double integ_fx[QUASOARE_GRAD_NFLUXES_INPUTS];
    double integ_dfx[QUASOARE_GRAD_NFLUXES_INPUTS];
    /*... dM dS dmij[i=1,2,3] dsig[i=1,2,3] : total 8 doubles */
    double deriv_coefs[QUASOARE_GRAD_NDERIV];
    double dmuij[QUASOARE_GRAD_NFLUXES_INPUTS];
    double dsij[QUASOARE_GRAD_NFLUXES_INPUTS];

    /* we do not store the number of iterations
     * This is a dummy variable */
    int niter[1];

    double store;
    double gamma;
    double Qstars_timestep[QUASOARE_GRAD_NFLUXES_POTENTIAL];

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
            idx_m0 = nreservoirs * nfxo * nipmd;

            /* Initial condition */
            if(t == 0)
                store = x0;
            else {
                store = x1[idx_x - nreservoirs];
                dx0 = &(dx1[idx_dxp]);
            }

            /* Input data. Initialise to 0 if nan */
            Qstars_timestep[0] = Qstars[idx_q];
            if(isnan(Qstars_timestep[0]))
                Qstars_timestep[0] = 0.;

            Qstars_timestep[1] = Qstars[idx_q + 1];
            if(isnan(Qstars_timestep[1]))
                Qstars_timestep[1] = 0.;

            /* Baseline flux mapping index */
            idx_m0 = nfxp * nreservoirs * nfxo * res;

            /* Loops through upstream links */
            for(ulink = 0; ulink < nreservoirs * nfxo; ulink++) {
                idx_m = idx_m0 + ulink * nipmd;
                resup_id = input_mapping[idx_m];

                /* No more links ... we break */
                if(resup_id < 0)
                    break;

                resup_ofx_id = input_mapping[idx_m + 1];
                resdown_pfx_id = input_mapping[idx_m + 2];

                /* here we convert the weight using the scaling factor */
                input_mapping_weight = (double)(input_mapping[idx_m + 3])
                    / input_mapping_scaling_factor;

                idx_up_f = (nres_times_t + resup_id) * nfxo + resup_ofx_id;
                Qstars_timestep[resdown_pfx_id] += input_mapping_weight * fluxes[idx_up_f];
            }

            /* Integrate ode */
            ierr = c_grad_integrate(nalphas, nreservoirs, nvariables,
                                    &(alphas[idx_a]),
                                    timestep,
                                    gamma,
                                    input_mapping,
                                    Qstars_timestep,
                                    dQstars_ext,
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

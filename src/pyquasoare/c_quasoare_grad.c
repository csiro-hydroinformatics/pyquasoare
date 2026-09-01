#include "c_quasoare_grad.h"

/* Solution of dS/dt = f*(s) */

double wfun(double delta, double S) {
    double x = delta * S;
    if(x > 460.51)
        return 1e200;

    if(x < -460.51)
        return 1e-200;

    return exp(x);
}

double c_lin_expmxy(double x, double y) {
    double z = x * y;
    if(isnull(x) || isnull(y))
        return x;
    return x * (expm1(z) / z);
}

double c_grad_forward(double M, double S, double t0, double x0, double t){
    double x = c_get_nan();
    double tau = t - t0;
    double exparg = 0.;

    if(isequal(t, t0, QUASOARE_EPS, 0.))
        return x0;

    if(isnull(S)){
        x = x0 + M * tau;
    }
    else {
        exparg = S * tau;
        if(fabs(exparg) < QUASOARE_EPS)
            x = x0 * (1 + exparg) + M * tau;
        else
            x = x0 * wfun(tau, S) + M * c_lin_expmxy(tau, S);
    }
    return x;
}


/* Primitive of 1/f*(s) */
double c_grad_inverse(double M, double S, double x0, double x){
    if(isnull(S)){
        return (x - x0) / M;
    }
    else {
        return 1. / S * log(fabs((S * x + M) / (S * x0 + M)));
    }
}


/* Integrate reservoir equation over 1 time step and compute associated fluxes
 * flux_scalings are scaling factors applied to
 * s[0] : f1 (inputs)
 * s[1] : f2 (demand)
 * The id of the variables is also matching this definition
 * id = 0 : derivative against reservoir
 * id = 1 : derivative of scaling applied to potential inflow
 * id = 2 : derivative of scaling applied to potential demand
 *
 * input_mapping : vector of nreservoirs × 6 output fluxes × nreservoirs × 2 input fluxes
 *   columns = [res1 / w(out1) w(out2) .. w(out6)] [res2 / w(out1) ..] ...
 *   rows = [res1 / (inp 1) (inp2)] [res2 / (inp1) (inp2)]
 *
 * dfluxes : vector of nvariables × 6 x nreservoirs
 * */
int c_grad_integrate(int nalphas, unsigned int nreservoirs, unsigned int nvariables,
                     unsigned int enforce_potential_inflow_pos,
                     double * alphas,
                     double timestep,
                     double gamma,
                     unsigned int *input_mapping,
                     double *input_mapping_weights,
                     double * Qstars,
                     double * dQstars_ext,
                     double * flux_scalings,
                     double * coefs_linear,
                     unsigned int current_reservoir_id,
                     unsigned int *variables_reservoir_id,
                     unsigned int *variables_type_id,
                     unsigned int *niter,
                     double * x,
                     double * dx,
                     double * fluxes,
                     double * dfluxes) {
    unsigned int ierr;
    unsigned int inp, out, dvar;
    unsigned long idx_c, idx_dq0, idx_dq;
    unsigned long idx_df, idx_df0;
    unsigned int nit=0, jalpha_next=0;

    unsigned int nfxi = QUASOARE_GRAD_NFLUXES_INPUTS;
    unsigned int nfxp = QUASOARE_GRAD_NFLUXES_POTENTIAL;
    unsigned int nfxo = QUASOARE_GRAD_NFLUXES_OUTPUTS;
    unsigned int nipmd = QUASOARE_GRAD_INPUT_MAPPING_LASTDIM;
    unsigned int ncoefs = QUASOARE_GRAD_NPIECEWISE_PARAMS;

    unsigned int ifxi = QUASOARE_GRAD_IFLUX_INFLOW;
    unsigned int ifxd = QUASOARE_GRAD_IFLUX_DEMAND;
    unsigned int ifxdr = QUASOARE_GRAD_IFLUX_DRAINAGE;
    unsigned int ifxb = QUASOARE_GRAD_IFLUX_BYPASS;
    unsigned int ifxs = QUASOARE_GRAD_IFLUX_SHORTFALL;
    unsigned int ifxo = QUASOARE_GRAD_IFLUX_OUTFLOW;
    unsigned int ifx_original;

    double M = 0., S = 0.;
    double term0, term1, term01, xmax;
    double mij = 0., sij = 0.;
    double expmb = 0.;
    double flx;
    double funval = 0.;
    double funval_prev = 0.;
    double alpha0, alpha1;
    double alpha_min = alphas[0];
    double alpha_max = alphas[nalphas-1];

    double mj_vect[QUASOARE_GRAD_NFLUXES_INPUTS];
    double sj_vect[QUASOARE_GRAD_NFLUXES_INPUTS];

    /* Variables related to potential fluxes */
    double qi;
    double scaling;
    /* .. extended Qstars including discharge fluxes
     *    + rescaling applied */
    double rescaled_Qstars_ext[QUASOARE_GRAD_NFLUXES_INPUTS];
    rescaled_Qstars_ext[0] = flux_scalings[0] * Qstars[0];
    rescaled_Qstars_ext[1] = flux_scalings[1] * Qstars[1];
    rescaled_Qstars_ext[2] = -flux_scalings[2] * gamma / timestep;

    /* Variables related to gradients */
    unsigned int reservoir_id, type_id;
    double w, R, dR, dS, dM, tau;
    double dx_start, dx_end;
    double integ_x, integ_dx;
    double integ_fx[QUASOARE_GRAD_NFLUXES_INPUTS];
    double integ_dfx[QUASOARE_GRAD_NFLUXES_INPUTS] = {0.};

    /* dmuij and dsij are used but never changed to other than 0 values.
     * This is a placeholder for future code to include derivatives against
     * linear approx coefficients */
    double dmuij[QUASOARE_GRAD_NFLUXES_INPUTS] = {0.};
    double dsij[QUASOARE_GRAD_NFLUXES_INPUTS] = {0.};

    double H, G;

    /* Max number of iteration
     * If s0 stays in [alpha0, alpha1], this number
     * should be <=nalphas. However we allow more iterations
     * if there is extrapolation */
    unsigned int niter_max = nalphas < 5 ? 10 : 2 * nalphas;

    /* Inialise variables */
    unsigned int extrapolating = 0;
    unsigned int extrapolating_low = 0;
    unsigned int extrapolating_high = 0;
    double x_low = 0;
    double x_high = 0;
    double t_final = timestep;
    double t_start = 0.;
    double t_end = 0.;
    double x_start = x[0];
    double x_end = x[0];

    /* Initial band */
    int jalpha = c_find_alpha(nalphas, alphas, x_start);

    /* Check data */
    qi = Qstars[0];

    if(enforce_potential_inflow_pos) {
        if(qi < -QUASOARE_EPS)
            return QUASOARE_NEGATIVE_POTENTIAL_INFLOW_FLUX;

        Qstars[0] = qi < 0 ? 0. : qi;
    }

    if(gamma <= 0)
        return QUASOARE_CAPACITY_IS_NEGATIVE_OR_NULL;

    /* Time loop */
    while (t_end < t_final * (1 - QUASOARE_EPS) && nit < niter_max) {
        //fprintf(stdout, "\n\t=> quasoare iteration %d\n", nit);
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

            /* Store flux coefficents */
            mj_vect[inp] = mij;
            sj_vect[inp] = sij;

            /* Add coefficients to obtain approximation of reservoir equation */
            qi = rescaled_Qstars_ext[inp] / gamma;
            M += mij * qi;
            S += sij * qi;

            //fprintf(stdout, "\n\t\tInput %d: sc=%3.3e mij=%3.3e qi=%3.3e g=%3.3e M = %3.3e",
            //        inp, flux_scalings[inp], coefs_linear[idx_c],
            //        Qstars_ext[inp], gamma, M);
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
            if(notequal(funval_prev, funval, QUASOARE_ATOL, QUASOARE_RTOL))
                return QUASOARE_NOT_CONTINUOUS;
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
                /* Ensure that x_end remains inside unsigned interpolation range */
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

        /* Compute the integral of x(t) over the time interval
         * See c_quasoare_core.c::c_quad_fluxes_increment
         * for explanation of this code */
        xmax = fabs(x_start) < fabs(x_end) ? fabs(x_end) : fabs(x_start);
        term0 = fabs(M);
        term1 = fabs(S * xmax);
        term01 = term0 < term1 ? term1 : term0;

        if(term0 * QUASOARE_FLUX_TERM_RTOL1 > term1 || isnull(term1)
                || term01 < QUASOARE_EPS) {
            integ_x = (x_start + M * tau / 2) * tau;
        }
        else {
            expmb = c_lin_expmxy(tau, S);
            integ_x = x_start * expmb + (expmb - tau) * (M / S);
        }
        if(isnan(integ_x))
            return QUASOARE_NAN_INTEGX;

        /* Increment fluxes during the last unsigned interval */
        for(inp = 0; inp < nfxi; inp++){
            /* Potential flux including flux scaling */
            qi = rescaled_Qstars_ext[inp];

            /* Compute fi(x) integral */
            integ_fx[inp] = sj_vect[inp] * integ_x + mj_vect[inp] * tau;
            flx = qi / timestep * integ_fx[inp];

            if(isnan(flx))
                return QUASOARE_NAN_FLUX;

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
            if(inp == ifxi || inp == ifxd) {
                if(nit == 1)
                    fluxes[inp + nfxi] = flx - qi / timestep * tau;
                else
                    fluxes[inp + nfxi] += flx - qi / timestep * tau;
            }

        }

        /* Total outflow = bypasss + drainage */
        fluxes[ifxo] = fluxes[ifxb] + fluxes[ifxdr];

        /******* Computation of sensitivities *******/

        /* Useful constants */
        w = wfun(tau, S);
        R = M / S;

        /* Loop over independent variables */
        for(dvar = 0; dvar < nvariables; dvar++) {
            //fprintf(stdout, "\t\tCompute sensitivities dvar=%d\n", dvar);
            reservoir_id = variables_reservoir_id[dvar];
            type_id = variables_type_id[dvar];

            /* Baseline index for derivative of Qstars */
            idx_dq0 = nfxi * dvar;

            /* Baseline index for flux derivatives */
            idx_df0 = nfxo * nreservoirs * dvar;

            /* Derivative of M, S against η */
            if (current_reservoir_id != reservoir_id) {
                /* -- variable belongs to a different reservoir -- */
                dM = (mj_vect[0] * dQstars_ext[idx_dq0]
                      + mj_vect[1] * dQstars_ext[idx_dq0 + 1]) / gamma;

                dS = (sj_vect[0] * dQstars_ext[idx_dq0]
                      + sj_vect[1] * dQstars_ext[idx_dq0 + 1]) / gamma;

            } else {
                /* -- variable belongs to the same reservoir -- */
                if (type_id == QUASOARE_GRAD_VARTYPE_CAPACITY) {
                    /* >> Gradient against capacity of the current reservoir << */
                    dM = -(rescaled_Qstars_ext[0] * mj_vect[0]
                        + rescaled_Qstars_ext[1] * mj_vect[1]) / gamma / gamma;

                    dS = -(rescaled_Qstars_ext[0] * sj_vect[0]
                        + rescaled_Qstars_ext[1] * sj_vect[1]) / gamma / gamma;

                    /* Derivative of Qstars */
                    dQstars_ext[idx_dq0] = 0.;
                    dQstars_ext[idx_dq0 + 1] = 0.;
                    /* .. dQ3* / dΓ = -η3 / Δ (don't forget rescaling!) */
                    dQstars_ext[idx_dq0 + 2] = rescaled_Qstars_ext[2] / gamma;
                }
                else if (type_id == QUASOARE_GRAD_VARTYPE_SCALING_INFLOW ||
                        type_id == QUASOARE_GRAD_VARTYPE_SCALING_DEMAND ||
                        type_id == QUASOARE_GRAD_VARTYPE_SCALING_DRAINAGE) {
                    /* >> Gradient against potentiel flux scaling << */

                    /* Scaling flux number is assumed to be type of variable - 1 */
                    inp = type_id - 1;

                    scaling = flux_scalings[inp];
                    if(isnull(scaling))
                        return QUASOARE_GRAD_ZERO_FLUX_SCALING;

                    mij = mj_vect[inp];
                    sij = sj_vect[inp];

                    qi = rescaled_Qstars_ext[inp] / scaling;
                    dM = qi * mij / gamma;
                    dS = qi * sij / gamma;

                    /* Derivative of Qstars : all null except inp */
                    dQstars_ext[idx_dq0] = 0.;
                    dQstars_ext[idx_dq0 + 1] = 0.;
                    dQstars_ext[idx_dq0 + 2] = 0.;
                    dQstars_ext[idx_dq0 + inp] = qi;
                }
                else
                    return QUASOARE_GRAD_TYPEID_NOT_FOUND;
            }

            /* Initialise dx */
            dx_start = dx[dvar];

            /* --- computation of ξ, integral of ξ, and integral of df(x)/dη --- */
            if(notequal(x_start, x_end, QUASOARE_EPS, 0.)) {
                if(notnull(S)) {
                    /* Compute dR and then dx_end andand integ_dx assuming S ≠ 0 */
                    dR = (dM - R * dS) / S;
                    dx_end = w * dx_start + dR * (w - 1.) + (x_start + R) * tau * dS * w;
                    integ_dx = (dx_end - dx_start - dM * tau - dS * integ_x) / S;

                } else {
                    /* Compute dx_end and integ_dx assuming S=0 */
                    dx_end = dx_start + dM * tau;
                    integ_dx = dM * tau * tau / 2;
                }

                for(inp = 0; inp < nfxi; inp++)
                    integ_dfx[inp] = dmuij[inp] * tau + dsij[inp] * integ_x
                        + sj_vect[inp] * integ_dx;

            } else
                dx_end = dx_start;

            if(!isfinite(dx_end))
                return QUASOARE_GRAD_IS_INFINITE;

            if(isnan(dx_end))
                return QUASOARE_GRAD_IS_NAN;

            /* Compute the two terms G and H related to flux derivatives */
            for(out = 0; out < nfxo ; out++) {

                if(out < nfxi) {
                    /* Inflow, demand and drainage fluxes */
                    idx_dq = idx_dq0 + out;
                    G = dQstars_ext[idx_dq] / timestep * integ_fx[out];
                    H = rescaled_Qstars_ext[out] * integ_dfx[out] / timestep;

                } else if (out == ifxb || out == ifxs) {
                    /* Bypass and demand shortfall */
                    ifx_original = out == ifxb ? ifxi : ifxd;
                    idx_dq = idx_dq0 + ifx_original;
                    G = dQstars_ext[idx_dq]
                        * (integ_fx[ifx_original] - tau) / timestep;
                    H = rescaled_Qstars_ext[ifx_original] * integ_dfx[ifx_original] / timestep;

                } else if (out == ifxo) {
                    /* Total outflow  = drainage + bypass*/
                    G = (dQstars_ext[idx_dq0 + ifxdr] * integ_fx[ifxdr]
                        + dQstars_ext[idx_dq0 + ifxi] * (integ_fx[ifxi] - tau))/ timestep;
                    H = (rescaled_Qstars_ext[ifxdr] * integ_dfx[ifxdr]
                        + rescaled_Qstars_ext[ifxi] * integ_dfx[ifxi]) / timestep;
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
            dx[dvar] = dx_end;
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
    *x = x_end;
    *niter = nit;

    /* Convergence problem */
    if(notequal(t_final, t_end, QUASOARE_ATOL, QUASOARE_RTOL) && fabs(funval)>QUASOARE_EPS)
        return QUASOARE_NO_CONVERGENCE;

    return 0;
}


int c_grad_reservoir(int nalphas, unsigned int nval,
                     unsigned int errors,
                     unsigned int enforce_potential_inflow_pos,
                     double * alphas,
                     double timestep,
                     double gamma,
                     double * flux_scalings,
                     double x0,
                     double * Qstars,
                     double * coefs_linear,
                     unsigned int * niter,
                     double * x,
                     double * dx,
                     double * fluxes,
                     double * dfluxes) {
    unsigned int ierr, t, dvar, i, inp;
    unsigned long idx_dx, idx_f, idx_df, idx_dq;

    unsigned int nfxi = QUASOARE_GRAD_NFLUXES_INPUTS;
    unsigned int nfxp = QUASOARE_GRAD_NFLUXES_POTENTIAL;
    unsigned int nfxo = QUASOARE_GRAD_NFLUXES_OUTPUTS;

    /* configure model for a single reservoir */
    unsigned int current_reservoir_id = 0;

    /* .. only one reservoir with no input mapping from others */
    unsigned int nreservoirs = 1;
    unsigned int input_mapping[QUASOARE_GRAD_INPUT_MAPPING_LASTDIM] = {QUASOARE_GRAD_INPUT_MAPPING_MISSINGRESERVOIRID};
    double input_mapping_weights[1] = {0.};

    /* .. we assume that we compute the derivative
     * agains all possible variable types */
    unsigned int nvariables = QUASOARE_GRAD_VARTYPE_NUMBER;
    unsigned int variables_reservoir_id[QUASOARE_GRAD_VARTYPE_NUMBER];
    unsigned int variables_type_id[QUASOARE_GRAD_VARTYPE_NUMBER];

    double dQstars_ext[ QUASOARE_GRAD_VARTYPE_NUMBER*QUASOARE_GRAD_NFLUXES_INPUTS ];

    /* initialise derivatives of state and inputs */
    x[0] = x0;

    for(dvar = 0; dvar < nvariables; dvar++){
        variables_reservoir_id[dvar] = 0;
        variables_type_id[dvar] = dvar;
        dx[dvar] = 0.;
    }

    /* Time loop */
    for(t = 0; t < nval; t++) {
        /* indexes */
        idx_dx = nvariables * t;
        idx_f = nfxo * t;
        idx_df = nvariables * idx_f; // nvar × nfx_out × t

        /* Integrate ode */
        ierr = c_grad_integrate(nalphas, nreservoirs, nvariables,
                                enforce_potential_inflow_pos,
                                alphas, timestep,
                                gamma,
                                input_mapping,
                                input_mapping_weights,
                                &(Qstars[nfxp * t]),
                                dQstars_ext,
                                flux_scalings,
                                coefs_linear,
                                current_reservoir_id,
                                variables_reservoir_id,
                                variables_type_id,
                                &(niter[t]),
                                &(x[t]),
                                &(dx[idx_dx]),
                                &(fluxes[idx_f]),
                                &(dfluxes[idx_df]));

        /* Manage errors */
        if(ierr > 0) {
            niter[t] = -1;

            for(i = 0; i < nfxo; i++)
                fluxes[idx_f + i] = c_get_nan();

            for(dvar = 0; dvar < nvariables; dvar++)
                dx[idx_dx + dvar] = c_get_nan();

            if(errors == 1)
                return ierr;
        }

        /* Loop states and diff states */
        if(t < nval - 1) {
            x[t + 1] = x[t];

            for(dvar = 0; dvar < nvariables; dvar ++)
                dx[idx_dx + nvariables + dvar] = dx[idx_dx + dvar];
        }
    }

    return 0;
}


int c_grad_model(int nalphas, unsigned int nreservoirs, unsigned int nvariables, unsigned int nval,
                 unsigned int errors,
                 unsigned int enforce_potential_inflow_pos,
                 double * alphas,
                 double timestep,
                 double * gammas,
                 unsigned int *input_mapping,
                 double *input_mapping_weights,
                 double * flux_scalings,
                 double * x0,
                 unsigned int *variables_reservoir_id,
                 unsigned int *variables_type_id,
                 double *Qstars,
                 double *dQstars_ext,
                 double *coefs_linear,
                 double *x,
                 double *dx,
                 double *fluxes,
                 double *dfluxes) {

    unsigned int ierr, t, dvar, inp, out, res, resu;
    unsigned long idx_q, idx_c, idx_x, idx_xn, idx_f, idx_a, idx_dx, idx_dxn;
    unsigned long idx_m, idx_m0, idx_mw, idx_mw0, idx_up_f;
    unsigned long idx_df, idx_up_df, idx_fs, idx_dq;
    unsigned long nres_times_t;

    unsigned int nfxi = QUASOARE_GRAD_NFLUXES_INPUTS;
    unsigned int nfxp = QUASOARE_GRAD_NFLUXES_POTENTIAL;
    unsigned int nfxo = QUASOARE_GRAD_NFLUXES_OUTPUTS;
    unsigned int nipmd = QUASOARE_GRAD_INPUT_MAPPING_LASTDIM;
    unsigned int ncoefs = QUASOARE_GRAD_NPIECEWISE_PARAMS;

    //fprintf(stdout, "\n\nDimensions:\n");
    //fprintf(stdout, "nalphas     = %d\n", nalphas);
    //fprintf(stdout, "nreservoirs = %d\n", nreservoirs);
    //fprintf(stdout, "nvariables  = %d\n", nvariables);
    //fprintf(stdout, "nval        = %d\n\n", nval);
    //fprintf(stdout, "nfxi   = %d\n", nfxi);
    //fprintf(stdout, "nfxp   = %d\n", nfxp);
    //fprintf(stdout, "nfxo   = %d\n", nfxo);
    //fprintf(stdout, "nipmd  = %d\n", nipmd);
    //fprintf(stdout, "ncoefs = %d\n", ncoefs);

    /* Variables related to store */
    unsigned int current_reservoir_id;
    double gamma;
    double Qstars_timestep[QUASOARE_GRAD_NFLUXES_POTENTIAL];

    /* Variables related to upstream input mapping */
    unsigned int ulink, resup_id, resup_ofx_id, resdown_pfx_id;
    double input_weight;

    /* Variables related to gradients */
    double w, R, dR, dS, dM, tau;
    double dx_start, dx_end;
    double dQstars_timestep_ext[QUASOARE_GRAD_NFLUXES_INPUTS];
    double integ_x;
    double integ_fx[QUASOARE_GRAD_NFLUXES_INPUTS];
    double integ_dfx[QUASOARE_GRAD_NFLUXES_INPUTS];
    unsigned int dvar_res_id;
    unsigned int dvar_type_id;

    /* we do not store the number of iterations
     * This is a dummy variable */
    unsigned int niter[1];

    if(nreservoirs >= QUASOARE_GRAD_INPUT_MAPPING_MISSINGRESERVOIRID)
        return QUASOARE_GRAD_NRESERVOIRS_TOOLARGE;

    unsigned int missid = QUASOARE_GRAD_INPUT_MAPPING_MISSINGRESERVOIRID;

    /* Initialise storage and diff storage -> no initial diff storage */
    for(res = 0; res < nreservoirs; res++) {
        x[res] = x0[res];

        idx_dx = res * nvariables;
        for(dvar = 0; dvar < nvariables; dvar ++)
            dx[idx_dx + dvar] = 0.;
    }

    /* timestep loop */
    for(t = 0; t < nval; t++) {

        /* Variable used to speed up index computations */
        nres_times_t = nreservoirs * t;

        /* index of flux derivatives for current timestep */
        idx_df = nres_times_t * nfxo * nvariables;

        /* loop through reservoirs */
        for(res = 0; res < nreservoirs; res++) {
            //fprintf(stdout, "\n\n\n--- t=%d res=%d ---\n", t, res);

            current_reservoir_id = res;

            /* initialise the derivatives of the 3 inputs */
            for(dvar = 0; dvar < nvariables; dvar ++) {
                dvar_res_id = variables_reservoir_id[dvar];
                dvar_type_id = variables_type_id[dvar];

                if(dvar_res_id >= nreservoirs || dvar_res_id < 0)
                   return QUASOARE_GRAD_RESERVOIRID_OUTOFBOUNDS;

                for(inp = 0; inp < nfxi; inp++) {
                    idx_dq = nfxi * dvar + inp;
                    dQstars_ext[idx_dq] = 0;
                }
            }

            /* Get reservoir capacity */
            gamma = gammas[res];

            /* Data indexes */
            idx_q = (nres_times_t + res) * nfxp;
            idx_fs = res * nfxi;
            idx_c = nfxi * (nalphas - 1) * ncoefs * res;
            idx_a = nalphas * res;
            idx_x = nres_times_t + res;
            idx_dx = (nres_times_t + res) * nvariables;
            idx_f = (nres_times_t + res) * nfxo;
            idx_m0 = nreservoirs * nfxo * nipmd * res;
            idx_mw0 = nreservoirs * nfxo * res;

            /* .. next time step */
            idx_xn = nres_times_t + nreservoirs + res;
            idx_dxn = (nres_times_t + nreservoirs + res) * nvariables;

            //fprintf(stdout, "\t=> Index values\n");
            //fprintf(stdout, "\tidx_q   = %d\n", idx_q);
            //fprintf(stdout, "\tidx_fs  = %d\n", idx_fs);
            //fprintf(stdout, "\tidx_c   = %d\n", idx_c);
            //fprintf(stdout, "\tidx_a   = %d\n", idx_a);
            //fprintf(stdout, "\tidx_x   = %d\n", idx_x);
            //fprintf(stdout, "\tidx_dx  = %d\n", idx_dx);
            //fprintf(stdout, "\tidx_f   = %d\n", idx_f);
            //fprintf(stdout, "\tidx_m0  = %d\n", idx_m0);
            //fprintf(stdout, "\tidx_xn  = %d\n", idx_xn);
            //fprintf(stdout, "\tidx_dxn = %d\n", idx_dxn);
            //fprintf(stdout, "\tidx_df  = %d\n", idx_df);

            /* Input data. Initialise to 0 if nan */
            Qstars_timestep[0] = Qstars[idx_q];
            if(isnan(Qstars_timestep[0]))
                Qstars_timestep[0] = 0.;

            Qstars_timestep[1] = Qstars[idx_q + 1];
            if(isnan(Qstars_timestep[1]))
                Qstars_timestep[1] = 0.;

            /* Loops through upstream links to compute input
             * and derivatives of input for the current reservoir */
            //fprintf(stdout, "\t=> Get upstream fluxes\n");
            for(ulink = 0; ulink < nreservoirs * nfxo; ulink++) {
                idx_m = idx_m0 + ulink * nipmd;
                resup_id = input_mapping[idx_m];

                /* No more links ... we break */
                if(resup_id == missid)
                    break;

                 resup_ofx_id = input_mapping[idx_m + 1];
                 if(resup_ofx_id >= QUASOARE_GRAD_NFLUXES_OUTPUTS)
                    return QUASOARE_GRAD_FLUXID_OUTOFBOUNDS;

                 resdown_pfx_id = input_mapping[idx_m + 2];
                 if(resdown_pfx_id >= QUASOARE_GRAD_NFLUXES_POTENTIAL)
                    return QUASOARE_GRAD_FLUXID_OUTOFBOUNDS;

                /* input weight */
                idx_mw = idx_mw0 + ulink;
                input_weight = input_mapping_weights[idx_mw];

                /* Index of upstream flux */
                idx_up_f = (nres_times_t + resup_id) * nfxo + resup_ofx_id;

                /* Affect flux */
                Qstars_timestep[resdown_pfx_id] += input_weight * fluxes[idx_up_f];

                //fprintf(stdout, "\t\t(R%d,F%d) -> (R%d,F%d) w=%3.3e x fx=%3.3e = %3.3e\n",
                //        resup_id, resup_ofx_id, res, resdown_pfx_id,
                //        input_weight, fluxes[idx_up_f],
                //        Qstars_timestep[resdown_pfx_id]);

                /* Derivative of Qstars */
                for(dvar = 0; dvar < nvariables; dvar++) {
                    dvar_res_id = variables_reservoir_id[dvar];
                    dvar_type_id = variables_type_id[dvar];

                    /* Locate the output flux derivative from the
                     * upstream reservoir */
                    idx_up_df = idx_df + nfxo  * (resup_id + nreservoirs * dvar) + resup_ofx_id;

                    /* Locate the Qstars derivative */
                    idx_dq = nfxi * dvar + resdown_pfx_id;

                    if(dvar_res_id < current_reservoir_id) {
                        /* Variable belongs to a reservoir located upstream.
                         * Compute dQst / dη based on upstream flux derivatives
                         * and their mapping to the current reservoir
                         *
                         * Note: this works if dfluxes have been computed
                         * for upstream reservoirs first. Hence the reservoir
                         * order needs to be upstream -> downstream
                         */
                        dQstars_ext[idx_dq] += input_weight * dfluxes[idx_up_df];
                    }
                    /* Case where variable belongs to downstream reservoir
                     * is ignored because their flux derivatives
                     * do not contribute to the derivatives of
                     * potential flows. Case where variable belongs to
                     * same reservoir is dealt with in the integrate routine.
                     * */
                }
            }
            //fprintf(stdout, "\t\tQst[0] = %3.3e\n", Qstars_timestep[0]);
            //fprintf(stdout, "\t\tQst[1] = %3.3e\n", Qstars_timestep[1]);

            /* Integrate ode */
            //fprintf(stdout, "\t=> Integrate\n");
            ierr = c_grad_integrate(nalphas, nreservoirs, nvariables,
                                    enforce_potential_inflow_pos,
                                    &(alphas[idx_a]),
                                    timestep,
                                    gamma,
                                    input_mapping,
                                    input_mapping_weights,
                                    Qstars_timestep,
                                    dQstars_ext,
                                    &(flux_scalings[idx_fs]),
                                    &(coefs_linear[idx_c]),
                                    current_reservoir_id,
                                    variables_reservoir_id,
                                    variables_type_id,
                                    niter,
                                    &(x[idx_x]),
                                    &(dx[idx_dx]),
                                    &(fluxes[idx_f]),
                                    &(dfluxes[idx_df]));

            //fprintf(stdout, "\n\t=> fluxes t=%d res=%d ierr=%d",
            //        t, res, ierr);
            //for(out=0; out < nfxo; out++)
            //    fprintf(stdout, "\n\t\tQ[%d] = %3.3e", out, fluxes[idx_f + out]);

            //fprintf(stdout, "\n\t=> dfluxes t=%d res=%d", t, res);
            //for(int dvar=0; dvar < nvariables; dvar ++) {
            //    fprintf(stdout, "\n\t\tvar=%d", dvar);
            //    for(int res2=0; res2 < nreservoirs; res2++) {
            //        fprintf(stdout, "\n\t\t\tres=%d\n\t\t\t\t", res2);
            //        for(int o=0; o<nfxo; o++)
            //            fprintf(stdout, "%6.3e ", dfluxes[idx_df + nfxo * nreservoirs * dvar + nfxo  * res2 + o]);
            //    }
            //    fprintf(stdout, "\n\n");
            //}

            /* error management */
            if(ierr > 0) {
                for(out = 0; out < nfxo; out++)
                    fluxes[idx_f + out] = c_get_nan();

                for(dvar = 0; dvar < nvariables; dvar++) {
                    dx[idx_dx + dvar] = c_get_nan();

                    for(res = 0; res < nreservoirs; res ++) {
                        for(out = 0; out < nfxo; out ++)
                            dfluxes[idx_df + res * nfxo + out] = c_get_nan();
                    }
                }

                if(errors == 1)
                    return ierr;
            }

            /* Loop states and diff states */
            if(t < nval - 1) {
                x[idx_xn] = x[idx_x];

                for(dvar = 0; dvar < nvariables; dvar++)
                    dx[idx_dxn + dvar] = dx[idx_dx + dvar];
            }
        }
    }

    return 0;
}

#include "c_quasoare_core.h"

/* Approximation functions */

/* [exp(x*y) - 1] / y */
double c_quad_expmxy(double x, double y) {
    double z = x * y;
    return x * (expm1(z) / z);
}

/* log(1 - x*y) / y */
double c_quad_logpxy(double x, double y) {
    double z = x * y;
    if(z >= 1.0)
        return c_get_nan();

    return x * (log1p(-z) / z);
}

double c_quad_fun(double a, double b, double c, double s){
    return notnan(s) ? (a * s + b) * s + c : c_get_nan();
}

double c_quad_grad(double a, double b, double c, double s){
    return notnan(s) ? 2. * a * s + b : c_get_nan();
}

int c_quad_coefficients_tangent(double a, double b, double c, double sref, double coefs[3]) {
    coefs[0] = 0;
    coefs[1] = c_quad_grad(a, b, c, sref);
    coefs[2] = c_quad_fun(a, b, c, sref) - sref * coefs[1];
    return 0;
}

double c_quad_fun_tangent(double a, double b, double c, double sref, double s){
    double coefs[3];
    c_quad_coefficients_tangent(a, b, c, sref, coefs);
    return c_quad_fun(coefs[0], coefs[1], coefs[2], s);
}

/*
 * Quadratic interpolation coefficients to match a function such that
 * f(a0) = f0 , f(a1) = f1 , f((a0+a1)/2) = fm
 * f(s) = as^2+bs+c
 * approx opt characterises the fit:
 * 0 = linear fit (i.e. force a = 0)
 * 1 = mononotonic function (i.e. prevent zero of derivative in [a0, a1])
 * 2 = free
 * */
int c_quad_coefficients(int approx_opt, double a0, double a1,
                        double f0, double f1, double fm,
                        double coefs[3]){
     double da = a1 - a0;
     double da2 = da * da;
     double A = 0, B = 0;
     double a = 0, b = 0, c = 0;

     if(a1<=a0){
        coefs[0] = c_get_nan();
        coefs[1] = c_get_nan();
        coefs[2] = c_get_nan();
        return QUASOARE_NONINCREASING_NODES;
     }

     /* Ensures quadratic function remains monotone if fm */
     if(approx_opt == 1){
        double f25 = (3 * f0 + f1) / 4;
        double f75 = (f0 + 3 * f1) / 4;
        double bnd1 = f25 < f75 ? f25 : f75;
        double bnd2 = f25 < f75 ? f75 : f25;
        fm = fm < bnd1 ? bnd1 : fm > bnd2 ? bnd2 : fm;
     }

     /* Linear function */
     if(approx_opt == 0) {
         a = 0.;
         b = (f1 - f0) / da;
         c = f0 - a0 * b;
     } else {
         /* Quadratic function */
         A = 2 * f0 + 2 * f1 - 4 * fm;
         B = 4 * fm - f1 - 3 * f0;

         a = A / da2;
         b = -2 * a0 * a + B / da;
         c = a0 * a0 * a - B * a0 / da + f0;
     }

     coefs[0] = a;
     coefs[1] = b;
     coefs[2] = c;

     return 0;
}

int c_quad_coefficients_smooth(double beta0, double beta1,
                               double f0, double f1,
                               double df0, double df1,
                               double coefs1[3], double coefs2[3]) {
    if(beta1 <= beta0) {
        coefs1[0] = c_get_nan();
        coefs1[1] = c_get_nan();
        coefs1[2] = c_get_nan();
        coefs2[0] = c_get_nan();
        coefs2[1] = c_get_nan();
        coefs2[2] = c_get_nan();
        return QUASOARE_NONINCREASING_NODES;
    }
    /* Compute two sets of coefficients that performs
     * a continuous piecewise quadratic interpolation
     * with values in beta0 and beta1 equal to f0 and f1,
     * derivatives equal to df0 and df1. The interpolation
     * relies on a basis of 4 piecewise quadratic functions
     * defined in [0, 1]. Each function is defined by two sets
     * of coefficients:
     * g1 : [  -2, 0, 1] / [  2, -4, 2]
     * g2 : [   2, 0, 0] / [ -2,  4, -1]
     * g3 : [-1.5, 1, 0] / [0.5, -1, 0.5]
     * g4 : [-0.5, 0, 0] / [1.5, -2, 0.5]
     *
     * To translate a function defined for x in [beta0, beta1]
     * to [0, 1], we introduce u = (x - beta0) / (beta1 - beta0) = m + s x
     * This leads to
     * a u^2 + b u + c = a (m + s x)^2 + b (m + s x) + c
     *                 = a s^2 x^2 + (b s + 2 a m s) x + a m^2 + b m + c
     * Hence
     * a -> a s^2
     * b -> b s + 2 a m s
     * c -> a m^2 + b m + c
     *
     * Also, we have x = (u - m) / s hence
     * df/du = df/dx 1/s
     */
    double delta = beta1 - beta0;
    double s = 1./delta;
    double m = -beta0 * s;
    double s_sq = s * s;
    double m_sq = m * m;

    df0 *= delta;
    df1 *= delta;

    double a1 = -2 * f0 + 2 * f1 - 1.5 * df0 - 0.5 * df1;
    double b1 = df0;
    double c1 = f0;
    coefs1[0] = a1 * s_sq;
    coefs1[1] = b1 * s + 2 * a1 * m * s;
    coefs1[2] = a1 * m_sq + b1 * m + c1;

    double a2 = 2 * f0 - 2 * f1 + 0.5 * df0 + 1.5 * df1;
    double b2 = -4 * f0 + 4 * f1 - df0 - 2 * df1;
    double c2 = 2 * f0 - f1 + 0.5 * df0 + 0.5 * df1;
    coefs2[0] = a2 * s_sq;
    coefs2[1] = b2 * s + 2 * a2 * m * s;
    coefs2[2] = a2 * m_sq + b2 * m + c2;

    return 0;
}


/* solution valididty range */
double c_quad_delta_t_max(double a, double b, double c,
                          double Delta, double qD, double sbar,
                          double s0){
    double delta_tmax = 0.;
    double tmp;

    if(isnull(a)){
        delta_tmax = c_get_inf();
    }
    else{
        tmp = a * (s0 - sbar);

        if(isnull(Delta))
            delta_tmax = tmp <= 0 ? c_get_inf() : 1. / tmp;

        else if (Delta<0)
            delta_tmax = (QUASOARE_PI / 2 - c_eta_fun(tmp / qD, Delta)) / qD;

        else if (Delta>0)
            delta_tmax = tmp < qD ? c_get_inf() : -c_eta_fun(tmp / qD, Delta) / qD;
    }
    return delta_tmax;
}


/* Solution of dS/dt = f*(s) */
double c_quad_forward(double a, double b, double c, double Delta, double qD,
                            double sbar, double t0, double s0, double t){
    double s1 = c_get_nan();
    double omega = 0.;
    double exparg = 0;
    double tau = t - t0;
    double signD = Delta < 0 ? -1. : 1.;
    double expmb = 0;

    /* Obtain the maximum time for which integration remains valid */
    double delta_tmax = c_quad_delta_t_max(a, b, c, Delta, qD, sbar, s0);

    if(t < t0 || t > t0 + delta_tmax)
        return s1;

    if(isequal(t, t0, QUASOARE_EPS, 0.))
        return s0;

    if(isnull(a) && isnull(b)){
        s1 = s0 + c * tau;
    }
    else if((isnull(a) && notnull(b))){
        exparg = b * tau;
        if(fabs(exparg) < QUASOARE_EPS) {
            s1 = s0 * (1 + exparg) + c * tau;
        } else {
            /* Computes exp(tau * b) - 1) / b */
            expmb = c_quad_expmxy(tau, b);
            s1 = s0 * exp(exparg) + expmb * c;
        }
    }
    else{
        if(isnull(Delta)){
            s1 = sbar + (s0 - sbar) / (1 - a * tau * (s0 - sbar));
        }
        else {
            omega = c_omega_fun(qD*tau, Delta);
            s1 = sbar + (s0 - sbar - signD * qD / a * omega)
                        / (1. - a / qD * (s0 - sbar) * omega);
        }
    }

    return s1;
}


/* Primitive of 1/f*(s) */
double c_quad_inverse(double a, double b, double c, double Delta, double qD,
                      double sbar, double s0, double s1){
    if(isnull(a) && isnull(b)){
        return (s1-s0)/c;
    }
    else if(isnull(a) && notnull(b)){
        return 1./b*log(fabs((b*s1+c)/(b*s0+c)));
    }
    else{
        if(isnull(Delta))
            return (1./(s0-sbar)-1./(s1-sbar))/a;
        else{
            return (c_eta_fun(a *(s1 - sbar) / qD, Delta)
                    - c_eta_fun(a * (s0 - sbar) / qD, Delta)) / qD;
        }
    }
}


/* Increment fluxes by integrating f*(s) */
int c_quad_fluxes_increment(int nfluxes,
                            double * aj_vector,
                            double * bj_vector,
                            double * cj_vector,
                            double Aj, double Bj, double Cj,
                            double Delta, double qD, double sbar,
                            double t0, double t1, double s0, double s1,
                            double * fluxes){
    int i, is_lin;
    double tau = t1 - t0;
    double aij, bij, cij;
    double a = Aj, a_check = 0.;
    double b = Bj, b_check = 0.;
    double c = Cj, c_check = 0.;

    /* Integration variables */
    double term0, term1, term2, term12;
    double c_b_ratio;
    double expmb = 0., expm2b = 0.;
    double y0 = 0., y1 = 0., l0 = 0., l1 = 0.;
    double integS = c_get_nan();
    double integS2 = c_get_nan();
    double smax = fabs(s0) < fabs(s1) ? fabs(s1) : fabs(s0);

    /* check integration time */
    if(t1 < t0)
        return QUASOARE_TIME_TOOLOW;

    if(isnull(tau))
        return 0;

    /* Decide on computation method depending on order of
     * magnitude of each terms of the equation dx/dt = ax^2 + bx + c */
    term0 = fabs(c);
    term1 = fabs(b * smax);
    term2 = fabs(a * smax * smax);
    term12 = term1 > term2 ? term1 : term2;

    /* Calculate integral of S and S^2 */
    if (term0 * QUASOARE_FLUX_TERM_RTOL > term12 || isnull(fabs(s1 - s0))) {
        /* Constant case where dx/dt = c */
        integS = (s0 + c * tau / 2) * tau;
        integS2 = (s0 * s0 + s0 * c * tau + c * c * tau * tau / 3.) * tau;
    }
    else if (term1 * QUASOARE_FLUX_TERM_RTOL > term2) {
        /* Linear case where dx/dt = bx + c */

        /* Rewrite of Eq B11 in Lerat (2025) for improved numerics when a≈0 */
        c_b_ratio = c / b;
        expmb = c_quad_expmxy(tau, b); // (exp(b * tau) - 1) / b
        integS = s0 * expmb + (expmb - tau) * c_b_ratio;

        /* Rewrite of Eq B12 in Lerat (2025) for improved numerics when a≈0 */
        expm2b = c_quad_expmxy(tau, 2 * b);
        integS2 = s0 * s0 * expm2b
            + c_b_ratio * c_b_ratio * (tau + expm2b - 2 * expmb)
            + 2 * s0 * c_b_ratio * (expm2b - expmb);
    }
    else {
        /* Quadratic case where dx/dt = ax^2 + bx + c */
        if(fabs(Delta) < QUASOARE_DELTA_ATOL) {
            /* Rewrite of Eq B14 in Lerat (2025) for better numerics */
            integS = sbar * tau - c_quad_logpxy(tau * (s0 - sbar), a);
            /* Eq B13 in Lerat (2025) */
            integS2 = (s1 - s0 - b * integS - c * tau) / a;
        }
        else {
            /* These formula use a change of variable such that
             * y = a x + b / 2
             *
             * Consequently we obtain:
             * dy/dt = y^2 - Δ / 4
             * where Δ = b^2 - 4ac
             *
             * Subsequently, we introduce f(y) = log(|y^2 - Δ/4|)
             * This leads to
             * df(y)/dt = 2 y
             *
             * Hence
             * ∫ydt = (f(y1) - f(y0)) / 2
             * => ∫x = [f(y1) - f(y0) - b(t1 - t0)] / 2a
             *
             * Similarly
             * ∫y^2 dt = y1 - y0 => a^2 ∫x^2 +ab ∫x + b^2/4 (t1 - t0) = y1 - y0
             */

            /* y variable at t=t0 */
            y0 = a * s0 + b / 2;
            l0 = log(fabs(y0 * y0 - Delta / 4));

            /* y variable at t = t1 */
            y1 = a * s1 + b / 2;
            l1 = log(fabs(y1 * y1 - Delta / 4));

            /*.. the isfinite clause should not be needed in theory,
             * but it seems to help a lot some edge cases??? */
            if(!isfinite(l1) || fabs(y1 * y1 - Delta) < QUASOARE_STEADY_STATE_TOL) {
                /* We are infinitely close to the steady state of x, which breaks
                 * the normal calculation of l1. Because there is a steady state,
                 * we assume Delta > 0. In this case, the only stable steady
                 * state is yb = -√Δ/2.
                 * Moreover, we know the solution for y, which leads to:
                 * (y1 + √Δ/2) / (y1 - √Δ/2) = (y0 + √Δ/2) / (y0 - √Δ/2) exp(-√Δ (t1 - t0))
                 *
                 * If we assume that y1 ≈ -√Δ/2, we obtain
                 * y1 ≈ -√Δ/2 - √Δ (y0 + √Δ/2) / (y0 - √Δ/2) exp(-√Δ (t1 - t0))
                 *
                 * A first order approximation gives
                 * |y1**2 - Δ/4| ≈ Δ |(y0 + √Δ/2) / (y0 - √Δ/2)| exp(-√Δ (t1 - t0))
                 *
                 * Hence
                 * log(|y1**2 - Δ/4|) ≈ log(Δ) + log(|(y0 + √Δ/2) / (y0 - √Δ/2)|) -√Δ (t1 - t0)
                 */
                l1 = log(Delta) + log(fabs((y0 + qD) / (y0 - qD))) - 2 * qD * tau;
            }

            integS = (l1 - l0 - b * tau) / 2 / a;
            integS2 = (y1 - y0 + b * (l0 - l1) / 2 + (b * b + Delta) * tau / 4) / a / a;
        }
    }

    /* increment fluxes */
    for(i = 0; i < nfluxes; i++){
        aij = aj_vector[i];
        a_check += aij;

        bij = bj_vector[i];
        b_check += bij;

        cij = cj_vector[i];
        c_check += cij;

        fluxes[i] += aij * integS2 + bij * integS + cij * tau;
    }

    if(notequal(a, a_check, QUASOARE_ATOL, QUASOARE_RTOL) ||
            notequal(b, b_check, QUASOARE_ATOL, QUASOARE_RTOL) ||
            notequal(c, c_check, QUASOARE_ATOL, QUASOARE_RTOL)){
        return QUASOARE_FAILEDSUMCHECK;
    }
    return 0;
}


/* Integrate reservoir equation over 1 time step and compute associated fluxes */
int c_quad_integrate(int nalphas, int nfluxes,
                            double * alphas, double * scalings,
                            double * coefs_noscaling,
                            double t0,
                            double s0,
                            double timestep,
                            int *niter, double * s1, double * fluxes) {
    int idx, i, nit=0, jalpha_next=0, err_flux;
    int ncoefs = 3;
    double Aj = 0., Bj = 0., Cj = 0.;
    double a=0., b=0., c=0.;
    double coefs_tangent[3];
    double constants[3], Delta, qD, sbar;
    double funval=0., funval_prev=0.;
    double alpha0, alpha1, scl;
    double alpha_min=alphas[0];
    double alpha_max=alphas[nalphas-1];

    double aj_vect[QUASOARE_NFLUXES_MAX];
    double bj_vect[QUASOARE_NFLUXES_MAX];
    double cj_vect[QUASOARE_NFLUXES_MAX];

    /* Max number of iteration
     * If s0 stays in [alpha0, alpha1], this number
     * should be <=nalphas. However we allow more iterations
     * if there is extrapolation */
    int niter_max = nalphas < 5 ? 10 : 2*nalphas;

    /* Initial band */
    int jalpha = c_find_alpha(nalphas, alphas, s0);

    /* Inialise variables */
    int extrapolating = 0;
    int extrapolating_low = 0;
    int extrapolating_high = 0;
    double s_low = 0;
    double s_high = 0;
    double t_final = t0 + timestep;
    double t_start = t0;
    double t_end = t0;
    double s_start = s0;
    double s_end = s0;

    if(nfluxes>=QUASOARE_NFLUXES_MAX)
        return QUASOARE_NFLUXES_TOO_LARGE;

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
        Aj=0.; Bj=0.; Cj=0.;

        for(i=0 ; i < nfluxes ; i++) {
            /* Scaling factor to be applied to each flux coefficient */
            scl = scalings[i];
            if(isnan(scl))
                return QUASOARE_NAN_SCALING;

            /* Compute flux coefficients
             * Multiply approximation coefficients by scaling.
             * if s is lower than alpha1 or higher than alpham
             * then set approx_fun to linear extrapolation
             * matching gradient at boundary.
             */
            if(extrapolating_low){
                idx = (nalphas - 1) * ncoefs * i;
                a = coefs_noscaling[idx] * scl;
                b = coefs_noscaling[idx + 1] * scl;
                c = coefs_noscaling[idx + 2] * scl;

                c_quad_coefficients_tangent(a, b, c, alpha_min, coefs_tangent);
                a = coefs_tangent[0];
                b = coefs_tangent[1];
                c = coefs_tangent[2];
            }
            else if(extrapolating_high){
                idx = (nalphas - 1) * ncoefs * i + ncoefs * (nalphas - 2);
                a = coefs_noscaling[idx] * scl;
                b = coefs_noscaling[idx + 1] * scl;
                c = coefs_noscaling[idx + 2] * scl;

                c_quad_coefficients_tangent(a, b, c, alpha_max, coefs_tangent);
                a = coefs_tangent[0];
                b = coefs_tangent[1];
                c = coefs_tangent[2];
            }
            else {
                idx = (nalphas - 1) * ncoefs * i + ncoefs * jalpha;
                a = coefs_noscaling[idx] * scl;
                b = coefs_noscaling[idx + 1] * scl;
                c = coefs_noscaling[idx + 2] * scl;
            }

            /* Store flux coefficents */
            aj_vect[i] = a;
            bj_vect[i] = b;
            cj_vect[i] = c;

            /* Add coefficients to obtain approximation of reservoir equation
             * function */
            Aj += a;
            Bj += b;
            Cj += c;
        }

        /* Check coefficients */
        if(isnan(Aj) || isnan(Bj) || isnan(Cj))
            return QUASOARE_NAN_COEFF;

        Aj = fabs(Aj) < QUASOARE_EPS ? 0. : Aj;
        Bj = fabs(Bj) < QUASOARE_EPS ? 0. : Bj;

        /* Compute discriminant variables */
        c_quad_constants(Aj, Bj, Cj, constants);
        Delta = constants[0];
        qD = constants[1];
        sbar = constants[2];

        /* Get derivative at beginning of time step */
        funval = c_quad_fun(Aj, Bj, Cj, s_start);

        /* Check continuity except for first iteration */
        if(nit > 1){
            if(notequal(funval_prev, funval, QUASOARE_ATOL, QUASOARE_RTOL)) {
                return QUASOARE_NOT_CONTINUOUS;
            }
        }

        /* Try integrating up to the end of the time step */
        if(fabs(funval)<QUASOARE_EPS)
            s_end = s_start;
        else
            s_end = c_quad_forward(Aj, Bj, Cj, Delta, qD, sbar,
                                    t_start, s_start, t_final);

        /* complete or move band if needed */
        if(s_end>=alpha0 && s_end<=alpha1) {
            /* .. s_end is within band => complete */
            t_end = t_final;
            jalpha_next = jalpha;
        }
        else {
            /* find next band depending if f is decreasing or non-decreasing */
            /* Increment time */
            if(extrapolating) {
                /* Ensure that s_end remains inside interpolation range */
                s_low = alpha_min + 2 * QUASOARE_EPS;
                s_high = alpha_max - 2 * QUASOARE_EPS;
                if(funval < 0 && extrapolating_high && s_end < s_high) {
                    s_end = s_high;
                    jalpha_next = nalphas-2;
                }
                else if (funval > 0 && extrapolating_low && s_end > s_low) {
                    s_end = s_low;
                    jalpha_next = 0;
                }
            }
            else {
                /* funval cannot be zero here because otherwise s_end
                 * is in [alpha0, alpha1] */
                if(funval < 0) {
                    s_end = alpha0;
                    jalpha_next = jalpha > -1 ? jalpha - 1 : -1;
                }
                else {
                    s_end = alpha1;
                    jalpha_next = jalpha >= nalphas - 2 ? nalphas - 1 : jalpha + 1;
                }
            }

            /* Increment time */
            t_end = t_start+c_quad_inverse(Aj, Bj, Cj,
                                           Delta, qD, sbar,
                                           s_start, s_end);
            t_end = t_end < t_final && fabs(funval) > QUASOARE_EPS ? t_end : t_final;
        }

        if(!isfinite(s_end))
            return QUASOARE_STORE_LEVEL_INFINITE;

        /* Increment fluxes during the last interval */
        err_flux = c_quad_fluxes_increment(nfluxes,
                                           aj_vect, bj_vect, cj_vect,
                                           Aj, Bj, Cj, Delta, qD, sbar,
                                           t_start, t_end, s_start, s_end, fluxes);
        if(err_flux>0)
            return err_flux;

        /* Loop for next band */
        funval_prev = c_quad_fun(Aj, Bj, Cj, s_end);

        if(isnull(funval_prev)) {
            /* .. f is null => we have reached steady state => complete */
            t_end = t_final;
            jalpha_next = jalpha;
        }
        else {
            t_start = t_end;
            s_start = s_end;
            jalpha = jalpha_next;
        }
    }

    /* Store results */
    *s1 = s_end;
    *niter = nit;

    /* Convergence problem */
    if(notequal(t_final, t_end, QUASOARE_ATOL, QUASOARE_RTOL) && fabs(funval)>QUASOARE_EPS)
        return QUASOARE_NO_CONVERGENCE;

    return 0;
}

/**
* Integrate reservoir equation over multiple time steps:
* - scalings [nval, nfluxes] : scalings applied to linear coefficients
* - other input args identical to c_integrate
* - s1 [nval] : final states
* - fluxes [nval, nfluxes] : flux computed
* - errors : 0=ignore, 1=raise, 2=warn
**/
int c_quad_model(int nalphas, int nfluxes, int nval, int errors, double timestep,
                 double * alphas, double * scalings,
                 double * perturb,
                 double * coefs_noscaling,
                 double s0,
                 double smin,
                 double smax,
                 int * niter,
                 double * s1, double * fluxes) {
    int ierr, t, j;
    double store = s0;
    double t0 = 0.;

    for(t=0; t<nval; t++) {
        store += perturb[t];

        /* Integrate ode */
        ierr = c_quad_integrate(nalphas, nfluxes, alphas,
                                &(scalings[nfluxes * t]),
                                coefs_noscaling,
                                t0, store, timestep,
                                &(niter[t]),
                                &(s1[t]),
                                &(fluxes[nfluxes * t]));

        /* Manage errors */
        if(ierr > 0) {
            if(errors == 1)
                return ierr;

            else if (errors == 0 || errors == 2){
                niter[t] = -1;

                s1[t] = store;

                for(j = 0; j < nfluxes; j++)
                    fluxes[nfluxes*t+j] = c_get_nan();
            }
        }

        /* Loop initial state */
        store = s1[t];

        /* Impose constraint */
        store = store < smin ? smin : store > smax ? smax : store;
        s1[t] = store;
    }

    return 0;
}

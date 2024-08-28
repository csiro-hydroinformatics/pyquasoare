#include "c_rezeq_quad.h"

/* Approximation functions */
double c_quad_fun(double a, double b, double c, double s){
    return (a*s+b)*s+c;
}

double c_quad_grad(double a, double b, double c, double s){
    return 2.*a*s+b;
}

int c_quad_steady(double a, double b, double c, double steady[2]){
    double q, x1, x2;
    double sign = b<0 ? -1. : 1.;
    double constants[2], Delta;

    c_quad_constants(a, b, c, constants);
    Delta = constants[0];

    steady[0] = c_get_nan();
    steady[1] = c_get_nan();

    if(notnull(a)){
        if(isnull(Delta)){
            steady[0] = -b/2./a;
        }
        else if(Delta>=0){
            q = -0.5*(b+sign*sqrt(Delta));
            x1 = q/a;
            x2 = c/q;
            steady[0] = x1<x2 ? x1 : x2;
            steady[1] = x1<x2 ? x2 : x1;
        }
    }
    else {
        if(notnull(b)){
            steady[0] = -c/b;
        }
    }
    return 0;
}

/*
 * Quadratic interpolation coefficients to match a function such that
 * f(a0) = f0 , f(a1) = f1 , f((a0+a1)/2) = fm
 * fapprox(s) = as^2+bs+c
 * approx opt characterise the fit:
 * 0 = linear fit (i.e. force a = 0)
 * 1 = mononotonous function (i.e. prevent zero of derivative in [a0, a1])
 * 2 = free
 * */
int c_quad_coefficients(int approx_opt, double a0, double a1,
                            double f0, double f1, double fm,
                            double coefs[3]){
     double da = a1-a0;
     double da2 = da*da;
     double A=0, B=0;
     double a=0, b=0, c=0;

     /* Ensures quadratic function remains monotone if fm */
     double f25=0., f75=0., bnd1=0., bnd2=0.;
     if(approx_opt==1){
        f25=(3*f0+f1)/4;
        f75=(f0+3*f1)/4;
        bnd1 = f25<f75 ? f25 : f75;
        bnd2 = f25<f75 ? f75 : f25;
        fm = fm<bnd1 ? bnd1 : fm>bnd2 ? bnd2 : fm;
     }

     if(isnull(da)){
        /* a0=a1, cannot interpolate */
        a = c_get_nan();
        b = c_get_nan();
        c = c_get_nan();
     }
     else {
        /* Linear function */
        if(approx_opt==0) {
            a = 0.;
            b = (f1-f0)/da;
            c = f0-a0*b;
        } else {
            /* Quadratic function */
            A = 2*f0+2*f1-4*fm;
            B = 4*fm-f1-3*f0;

            a = A/da2;
            b = -2*a0*a+B/da;
            c = a0*a0*a-B*a0/da+f0;
        }
     }
     coefs[0] = a;
     coefs[1] = b;
     coefs[2] = c;
     return 0;
}

/* solution valididty range */
double c_quad_delta_t_max(double a, double b, double c,
                            double Delta, double qD, double sbar,
                            double s0){
    double delta_tmax=0.;
    double tmp = a*(s0-sbar);

    if(isnull(a)){
        delta_tmax = c_get_inf();
    }
    else{
        if(isnull(Delta))
            delta_tmax = tmp<=0 ? c_get_inf() : 1./tmp;
        else if (Delta<0)
            delta_tmax = (REZEQ_PI/2-c_eta_fun(tmp/qD, Delta))/qD;
        else if (Delta>0)
            delta_tmax = tmp<qD ? c_get_inf() : -c_eta_fun(tmp/qD, Delta)/qD;
    }
    return delta_tmax;
}

/* Solution of dS/dt = f*(s) */
double c_quad_forward(double a, double b, double c, double Delta, double qD,
                            double sbar, double t0, double s0, double t){
    double s1=c_get_nan();
    double omega=0.;
    double exparg=0;
    double tau=t-t0;
    double signD = Delta<0 ? -1. : 1.;

    /* Obtain the maximum time for which integration remains valid */
    double delta_tmax = c_quad_delta_t_max(a, b, c, Delta, qD, sbar, s0);

    if(t<t0 || t>t0+delta_tmax)
        return s1;

    if(isequal(t, t0, REZEQ_EPS, 0.))
        return s0;

    if(isnull(a) && isnull(b)){
        s1 = s0+c*tau;
    }
    else if((isnull(a) && notnull(b))){
        exparg = b*tau;
        s1 = fabs(exparg)<REZEQ_EPS ? s0*(1+exparg)+c*tau : -c/b+(s0+c/b)*exp(exparg);
    }
    else{
        if(isnull(Delta)){
            s1 = sbar+(s0-sbar)/(1-a*tau*(s0-sbar));
        }
        else {
            omega = c_omega_fun(qD*tau, Delta);
            s1 = sbar+(s0-sbar-signD*qD/a*omega)/(1.-a/qD*(s0-sbar)*omega);
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
            return (c_eta_fun(a*(s1-sbar)/qD, Delta)
                                -c_eta_fun(a*(s0-sbar)/qD, Delta))/qD;
        }
    }
    return c_get_nan();
}


/* Increment fluxes by integrating f*(s) */
int c_quad_fluxes(int nfluxes,
                        double * aj_vector,
                        double * bj_vector,
                        double * cj_vector,
                        double aoj, double boj, double coj,
                        double Delta, double qD, double sbar,
                        double t0, double t1, double s0, double s1,
                        double * fluxes){
    int i;
    double tau=t1-t0;
    double tau2=tau*tau;
    double tau3=tau2*tau;
    double aij, bij, cij;
    double a = aoj, a_check=0.;
    double b = boj, b_check=0.;
    double c = coj, c_check=0.;
    double integS=0., integS2=0.;
    double omega=0., signD=0., term1=0., term2=0.;

    if(t1<t0)
        return REZEQ_QUAD_TIME_TOOLOW;

    /* Calculate integral of S and S^2 */
    if (isnull(a) && isnull(b)) {
        integS = s0*tau+c*tau2/2;
        integS2 = s0*s0*tau+s0*c*tau2+c*c*tau3/3.;
    }
    else if((isnull(a) && notnull(b))){
        integS = (s1-s0-coj*tau)/boj;
        integS2 = ((s1*s1-s0*s0)/2.-coj*integS)/boj;
    }
    else if (notnull(a)){
        /* Integrate S only because S2 can be deducted from S */
        if(isnull(Delta))
            integS = sbar*tau-log(1-a*tau*(s0-sbar))/a;
        else {
            omega = c_omega_fun(qD*tau, Delta);
            signD = Delta<0 ? -1. : 1.;

            /* Special formula to avoid the case where omega == 1 */
            if(qD*tau>10 && Delta>0)
                term1 = (log(2)-qD*tau)/a;
            else
                term1 = log(1-signD*omega*omega)/2./a;

            term2 = -log(1-a*(s0-sbar)/qD*omega)/a;
            integS = sbar*tau+term1+term2;
        }
    }

    /* increment fluxes */
    for(i=0; i<nfluxes; i++){
        aij = aj_vector[i];
        a_check += aij;

        bij = bj_vector[i];
        b_check += bij;

        cij = cj_vector[i];
        c_check += cij;

        if(isnull(a))
            fluxes[i] += aij*integS2+bij*integS+cij*tau;
        else
            fluxes[i] += aij/a*(s1-s0)+(bij-aij*b/a)*integS+(cij-aij*c/a)*tau;
    }

    if(notequal(a, a_check, REZEQ_ATOL, REZEQ_RTOL) ||
            notequal(b, b_check, REZEQ_ATOL, REZEQ_RTOL) ||
            notequal(c, c_check, REZEQ_ATOL, REZEQ_RTOL)){
        //fprintf(stdout, "a=%5.5e check=%5.5e diff=%5.5e\n", a, a_check, a-a_check);
        //fprintf(stdout, "b=%5.5e check=%5.5e diff=%5.5e\n", b, b_check, b-b_check);
        //fprintf(stdout, "c=%5.5e check=%5.5e diff=%5.5e\n", c, c_check, c-c_check);
        return REZEQ_QUAD_FAILEDSUMCHECK;
    }
    return 0;
}


/* Integrate reservoir equation over 1 time step and compute associated fluxes */
int c_quad_integrate(int nalphas, int nfluxes,
                            double * alphas, double * scalings,
                            double * a_matrix_noscaling,
                            double * b_matrix_noscaling,
                            double * c_matrix_noscaling,
                            double t0,
                            double s0,
                            double timestep,
                            int *niter, double * s1, double * fluxes) {
    int REZEQ_DEBUG=1;

    int i, nit=0, jalpha_next=0, err_flux;
    double aoj=0., boj=0., coj=0.;
    double a=0., b=0., c=0.;
    double constants[2], Delta, qD, sbar;
    double funval=0., funval_prev=0., grad=0.;
    double alpha0, alpha1;
    double alpha_min=alphas[0];
    double alpha_max=alphas[nalphas-1];

    double a_vect[REZEQ_NFLUXES_MAX];
    double b_vect[REZEQ_NFLUXES_MAX];
    double c_vect[REZEQ_NFLUXES_MAX];

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
    double t_final = t0+timestep;
    double t_start=t0, t_end=t0;
    double s_start=s0, s_end=s0;

    if(nfluxes>REZEQ_NFLUXES_MAX)
        return REZEQ_QUAD_NFLUXES_TOO_LARGE;

    for(i=0; i<nfluxes; i++)
        fluxes[i] = 0.;

    if(REZEQ_DEBUG==1){
        fprintf(stdout, "\n\nNALPHAS=%d  NFLUXES=%d\n", nalphas, nfluxes);
        fprintf(stdout, "scalings:");
        for(i=0; i<nfluxes; i++)
            fprintf(stdout, " scl[%d]=%3.3e", i, scalings[i]);

        fprintf(stdout, "\nStart integrate s0=%0.3f j=%d\n", s0, jalpha);
    }

    /* Time loop */
    while ((t_end<t_final) && nit<niter_max) {
        nit += 1;

        /* Extrapolation is triggered if s_start is
         * below alpha0+EPS for low and
         * above alpha1-EPS for high */
        extrapolating_low = jalpha<0 ? 1 : 0;
        extrapolating_high = jalpha>=nalphas-1 ? 1 : 0;
        extrapolating = extrapolating_low+extrapolating_high>0 ? 1 : 0;

        /* Get band limits */
        alpha0 = extrapolating_low ? -1*c_get_inf() : alphas[jalpha];
        alpha1 = extrapolating_high ? c_get_inf() : alphas[jalpha+1];

        /* Sum coefficients accross fluxes */
        aoj=0.; boj=0.; coj=0.;

        for(i=0;i<nfluxes;i++){
            /* Compute flux coefficients
             * Multiply approximation coefficients by scaling.
             * if s is lower than alpha1 or higher than alpham
             * then set approx_fun to linear extrapolation
             * matching gradient at boundary.
             */
            if(extrapolating_low){
                a = a_matrix_noscaling[i]*scalings[i];
                b = b_matrix_noscaling[i]*scalings[i];
                c = c_matrix_noscaling[i]*scalings[i];

                grad = c_quad_grad(a, b, c, alpha_min);
                c = c_quad_fun(a, b, c, alpha_min)-grad*alpha_min;
                b = grad;
                a = 0.;
            }
            else if(extrapolating_high){
                a = a_matrix_noscaling[nfluxes*(nalphas-2)+i]*scalings[i];
                b = b_matrix_noscaling[nfluxes*(nalphas-2)+i]*scalings[i];
                c = c_matrix_noscaling[nfluxes*(nalphas-2)+i]*scalings[i];

                grad = c_quad_grad(a, b, c, alpha_max);
                c = c_quad_fun(a, b, c, alpha_max)-grad*alpha_max;
                b = grad;
                a = 0.;
            }
            else {
                a = a_matrix_noscaling[nfluxes*jalpha+i]*scalings[i];
                b = b_matrix_noscaling[nfluxes*jalpha+i]*scalings[i];
                c = c_matrix_noscaling[nfluxes*jalpha+i]*scalings[i];
            }

            /* Store flux coefficents */
            a_vect[i] = a;
            b_vect[i] = b;
            c_vect[i] = c;

            /* Add coefficients to obtain approximation of reservoir equation
             * function */
            aoj += a;
            boj += b;
            coj += c;
        }
        if(isnan(aoj) || isnan(boj) || isnan(coj))
            return REZEQ_QUAD_NAN_COEFF;

        if(REZEQ_DEBUG==1){
            fprintf(stdout, "\nCoefs:\n");
            for(i=0; i<nfluxes; i++)
                fprintf(stdout, " av[%d]=%3.3e bv[%d]=%3.3e cv[%d]=%3.3e\n",
                                    i, a_vect[i], i, b_vect[i], i, c_vect[i]);
            fprintf(stdout, " aoj=%3.3e boj=%3.3e coj=%3.3e\n",
                            aoj, boj, coj);
        }

        /* Compute discriminant variables */
        c_quad_constants(aoj, boj, coj, constants);
        Delta = constants[0];
        qD = constants[1];
        sbar = constants[2];

        /* Get derivative at beginning of time step */
        funval = c_quad_fun(aoj, boj, coj, s_start);

        /* Check continuity except for first iteration */
        if(nit>1){
            if(notequal(funval_prev, funval_prev, REZEQ_EPS*1e2, 1e-5)) {
                if(REZEQ_DEBUG==1)
                    fprintf(stdout, "    jalpha=%d/%d "
                                "funval(%0.5f, %0.5f, %0.5f, %0.5f)=%5.5e\n"
                                "                 "
                                "funval_prev=%5.5e diff=%5.5e\n",
                                jalpha, nalphas-2, aoj, boj, coj,
                                s_start, funval, funval_prev,
                                fabs(funval-funval_prev));
                return REZEQ_QUAD_NOT_CONTINUOUS;
            }
        }

        /* Try integrating up to the end of the time step */
        s_end = c_quad_forward(aoj, boj, coj, Delta, qD, sbar,
                                    t_start, s_start, t_final);

        /* complete or move band if needed */
        if((s_end>=alpha0 && s_end<=alpha1 && 1-extrapolating) || isnull(funval)){
            /* .. s_end is within band => complete */
            t_end = t_final;
            jalpha_next = jalpha;

            if(isnull(funval))
                s_end = s_start;
        }
        else {
            /* find next band depending if f is decreasing or non-decreasing */
            if(funval<0) {
                s_end = alpha0;
                jalpha_next = jalpha>-1 ? jalpha-1 : -1;
            }
            else {
                s_end = alpha1;
                jalpha_next = jalpha>=nalphas-2 ? nalphas-1 : jalpha+1;
            }

            /* Increment time */
            if(extrapolating) {
                /* we also need to correct s_end because it is infinite */
                s_end = c_quad_forward(aoj, boj, coj, Delta, qD, sbar,
                                            t_start, s_start, t_final);
                /* Ensure that s_end remains inside interpolation range */
                if(funval<0 && extrapolating_high){
                    s_end = s_end < alpha_max-2*REZEQ_EPS ?
                                        alpha_max-2*REZEQ_EPS : s_end;
                }
                else if (funval>0 && extrapolating_low){
                    s_end = s_end > alpha_min+2*REZEQ_EPS ?
                                        alpha_min+2*REZEQ_EPS : s_end;
                }
            }
            t_end = t_start+c_quad_inverse(aoj, boj, coj, Delta, qD, sbar,
                                                    s_start, s_end);
            t_end = isinf(t_end) ? t_final : t_end;
        }

        if(REZEQ_DEBUG==1)
            fprintf(stdout, "\n{%d} low=%d high=%d / fun=%3.3e"\
                    "/ t=%3.3e>%3.3e / j=%d>%d / s=%3.3e>%3.3e\n",
                    nit, extrapolating_low, extrapolating_high,
                    funval, t_start, t_end, jalpha, jalpha_next,
                    s_start, s_end);

        /* Increment fluxes during the last interval */
        if(REZEQ_DEBUG==1){
            fprintf(stdout, "\nCoefs:\n");
            for(i=0; i<nfluxes; i++)
                fprintf(stdout, " av[%d]=%3.3e bv[%d]=%3.3e cv[%d]=%3.3e\n",
                                    i, a_vect[i], i, b_vect[i], i, c_vect[i]);
            fprintf(stdout, " aoj=%3.3e boj=%3.3e coj=%3.3e\n",
                            aoj, boj, coj);
        }


        err_flux = c_quad_fluxes(nfluxes,
                    a_vect, b_vect, c_vect,
                    aoj, boj, coj, Delta, qD, sbar,
                    t_start, t_end, s_start, s_end, fluxes);
        if(REZEQ_DEBUG==1)
            fprintf(stdout, "   err flux = %d\n", err_flux);

        if(err_flux>0)
            return err_flux;

        /* Loop for next band */
        funval_prev = c_quad_fun(aoj, boj, coj, s_end);
        t_start = t_end;
        s_start = s_end;
        jalpha = jalpha_next;
    }

    /* Store results */
    *s1 = s_end;
    *niter = nit;

    if(REZEQ_DEBUG==1)
        fprintf(stdout, "\nEnd integrate s1=%0.5f j=%d\n", s_end, jalpha);

    /* Convergence problem */
    if(t_final-t_end>0) {
        return REZEQ_QUAD_NO_CONVERGENCE;
    }
    return 0;
}

/**
* Integrate reservoir equation over multiple time steps:
* - scalings [nval, nfluxes] : scalings applied to linear coefficients
* - other input args identical to c_integrate
* - s1 [nval] : final states
* - fluxes [nval, nfluxes] : flux computed
**/
int c_quad_model(int nalphas, int nfluxes, int nval, double timestep,
                            double * alphas, double * scalings,
                            double * a_matrix_noscaling,
                            double * b_matrix_noscaling,
                            double * c_matrix_noscaling,
                            double s0, int * niter,
                            double * s1, double * fluxes) {
    int ierr, t;
    double t0=0.;

    for(t=0; t<nval; t++){
        ierr = c_quad_integrate(nalphas, nfluxes, alphas,
                            &(scalings[nfluxes*t]),
                            a_matrix_noscaling,
                            b_matrix_noscaling,
                            c_matrix_noscaling,
                            t0, s0, timestep,
                            &(niter[t]),
                            &(s1[t]),
                            &(fluxes[nfluxes*t]));
        if(ierr>0)
            return ierr;

        /* Loop initial state */
        s0 = s1[t];
    }

    return 0;
}

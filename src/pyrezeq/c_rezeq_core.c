#include "c_rezeq_core.h"

/* Approximation functions */
double c_approx_fun(double nu, double a, double b, double c, double s){
    if(nu<0 || isnan(nu))
        return c_get_nan();

    return a+b*exp(-nu*s)+c*exp(nu*s);
}

double c_approx_jac(double nu, double a, double b, double c, double s){
    if(nu<0 || isnan(nu))
        return c_get_nan();

    return -nu*b*exp(-nu*s)+nu*c*exp(nu*s);
}

/* Validity interval of solution to dS/dt=f*(S) */
double c_integrate_delta_t_max(double nu, double a, double b, double c,
                                double s0){
    double e0 = exp(-nu*s0);
    double Delta = a*a-4*b*c;
    double delta_tmax=0, tmp=0, lam0=0, sqD=0;

    if(nu<0 || isnan(nu))
        return c_get_nan();

    if(isnull(b) && isnull(c)){
        /* Constant -> always valid */
        delta_tmax = c_get_inf();
    }
    else if(isnull(a) && isnull(b) && notnull(c)){
        delta_tmax = c<0 ? c_get_inf() : e0/nu/c;
    }
    else if(isnull(a) && notnull(b) && isnull(c)){
        delta_tmax = b>0 ? c_get_inf() : -1/e0/nu/b;
    }
    else if(notnull(a) && isnull(b) && notnull(c)){
        delta_tmax = c<0 || (c>0 && a<-c/e0) ? c_get_inf() : log(1+a*e0/c)/nu/a;
    }
    else if(notnull(a) && notnull(b) && isnull(c)){
        delta_tmax = b>0 || (b<0 && a>-b*e0) ? c_get_inf() : -log(1+a/e0/b)/nu/a;
    }
    else if(notnull(b) && notnull(c)){
        sqD = sqrt(fabs(Delta));
        lam0 = (2*b*e0+a)/sqD;

        if(isnull(Delta)){
            delta_tmax = a<-2*e0*b ? -2/(a+2*b*e0)/nu : c_get_inf();
            delta_tmax = fmin(delta_tmax, a>-c/e0 ?
                                4*b*e0/(a+2*b*e0)/nu/a : c_get_inf());
        }
        else if (isneg(Delta)){
            delta_tmax = lam0<0 ? atan(-1./lam0)*2/nu/sqD : c_get_inf();
            tmp = atan((lam0*sqD-a)/(a*lam0+sqD))*2/nu/sqD;
            tmp = tmp>0 ? tmp : c_get_inf();
            delta_tmax = fmin(fmin(delta_tmax, tmp), REZEQ_PI/nu/sqD);
        }
        else {
            delta_tmax = lam0<-1 ? atanh(-1./lam0)*2/nu/sqD : c_get_inf();
            tmp = (lam0*sqD-a)/(a*lam0-sqD);
            tmp = fabs(tmp)<1 ? atanh(tmp)*2/nu/sqD : -c_get_inf();
            tmp = tmp>0 ? tmp : c_get_inf();
            delta_tmax = fmin(delta_tmax, tmp);
        }
    }
    return delta_tmax>=0 ? delta_tmax : c_get_nan();
}

/* Solution of dS/dt = f*(s) */
double c_integrate_forward(double nu, double a, double b, double c,
                        double t0, double s0, double t){
    double e0 = exp(-nu*s0);
    double sgn=1,omeg=0, lam0=0, sqD=0;
    double Delta = a*a-4*b*c;
    double ra2b, s1=0;

    if(nu<0 || isnan(nu))
        return c_get_nan();

    if(isneg(t-t0))
        return c_get_nan();

    if(isequal(t, t0))
        return s0;

    double dtmax = c_integrate_delta_t_max(nu, a, b, c, s0);
    if(t-t0>dtmax)
        return c_get_nan();

    if(isnull(b) && isnull(c)){
        s1 = s0+a*(t-t0);
    }
    else if(isnull(a) && isnull(b) && notnull(c)){
        s1 = s0-log(1-nu*c/e0*(t-t0))/nu;
    }
    else if(isnull(a) && notnull(b) && isnull(c)){
        s1 = s0+log(1+nu*b*e0*(t-t0))/nu;
    }
    else if(notnull(a) && isnull(b) && notnull(c)){
        s1 = s0+a*(t-t0)-log(1+c/a/e0*(1-exp(nu*a*(t-t0))))/nu;
    }
    else if(notnull(a) && notnull(b) && isnull(c)){
        s1 = s0+a*(t-t0)+log(1+b/a*e0*(1-exp(-nu*a*(t-t0))))/nu;
    }
    else if(notnull(b) && notnull(c)){
        ra2b = a/2/b;
        if(isnull(Delta)){
            /* Determinant is zero */
            s1 = -log((e0+ra2b)/(1+(e0+ra2b)*nu*b*(t-t0))-ra2b)/nu;
        }
        else {
            /* Non zero determinant */
            sgn = isneg(Delta) ? -1 : 1;
            sqD = sqrt(sgn*Delta);
            omeg = isneg(Delta) ? tan(nu*sqD*(t-t0)/2) : tanh(nu*sqD*(t-t0)/2);
            lam0 = (2*b*e0+a)/sqD;
            s1 = -log((lam0+sgn*omeg)/(1+lam0*omeg)*sqD/2/b-ra2b)/nu;
        }
    }
    return s1;
}

/* Primitive of 1/f*(s) */
double c_integrate_inverse(double nu, double a, double b, double c,
                                double s0, double s1){
    double e0 = exp(-nu*s0);
    double e1 = exp(-nu*s1);
    double Delta = a*a-4*b*c;
    double sgn = Delta<0 ? -1 : 1;
    double L=0, tau0=0, tau1=0, sqD=0., lam0=0., lam1=0.;

    if(nu<0 || isnan(nu))
        return c_get_nan();

    if(isnull(b) && isnull(c)){
        return (s1-s0)/a;
    }
    else if(isnull(a) && isnull(b) && notnull(c)){
        return -e1/nu/c+e0/nu/c;
    }
    else if(isnull(a) && notnull(b) && isnull(c)){
        return 1./e1/nu/b-1./e0/nu/b;
    }
    else if(notnull(a) && isnull(b) && notnull(c)){
        return log((c+a*e0)/(c+a*e1))/nu/a;
    }
    else if(notnull(a) && notnull(b) && isnull(c)){
        return log((b+a/e1)/(b+a/e0))/nu/a;
    }
    else if(notnull(b) && notnull(c)){
        sqD = sqrt(sgn*Delta);
        lam0 = (2*b*e0+a)/sqD;
        lam1 = (2*b*e1+a)/sqD;

        if(isnull(Delta)){
            /* Determinant is zero */
            tau0 = 2./(a+2*b*e0)/nu;
            tau1 = 2./(a+2*b*e1)/nu;
            return tau1-tau0;
        }
        else if (ispos(Delta)){
            L = (1+lam1)*(1-lam0)/(1-lam1)/(1+lam0);
            return L>0 ? 1./sqD/nu*log(L) : c_get_nan();
        }
        else {
            return -2./sqD/nu*(atan(lam1)-atan(lam0));
        }
    }
    return c_get_nan();
}


/* Increment fluxes by integrating f*(s) */
int c_increment_fluxes(int nfluxes, double nu,
                        double * aj_vector,
                        double * bj_vector,
                        double * cj_vector,
                        double aoj, double boj, double coj,
                        double t0, double t1, double s0, double s1,
                        double * fluxes){
    int i;
    double dt = t1-t0;
    double e0 = exp(-nu*s0);
    double expint=0;
    double a = aoj, a_check=0;
    double b = boj, b_check=0;
    double c = coj, c_check=0;
    double A, B, C;
    double Delta = aoj*aoj-4*boj*coj;
    double sqD, aij, bij, cij, lam0, u0, u1, w;

    if(t1<t0 || nu<0 || isnan(nu))
        return REZEQ_ERROR + __LINE__;

    /* Integrate exp(-nuS) if needed */
    if(notnull(b) || notnull(c)){
        if(isnull(a) && isnull(b) && notnull(c)){
            expint = dt*e0-nu*c/2*dt*dt;
        }
        else if(isnull(a) && notnull(b) && isnull(c)){
            expint = dt/e0+nu*b/2*dt*dt;
        }
        else if(notnull(a) && isnull(b) && notnull(c)){
            expint = (e0+c/a)/nu/a*(1-exp(-nu*a*dt))-c/a*dt;
        }
        else if(notnull(a) && notnull(b) && isnull(c)){
            expint = -(1/e0+b/a)/nu/a*(1-exp(nu*a*dt))-b/a*dt;
        }
        else if(notnull(b) && notnull(c)){
            sqD = sqrt(fabs(Delta));
            if(isnull(Delta)){
                expint = log(1+(e0+a/2/b)*nu*b*dt)/nu/b-a/2/b*dt;
            }
            else {
                lam0 = (2*b*e0+a)/sqD;
                if (ispos(Delta)){
                    w = nu*sqD/2*dt;
                    /* Care with overflow */
                    if(w>100) {
                        expint = (log((lam0+1)/2)+w)/nu/b-a/2/b*dt;
                    }
                    else {
                        u1 = exp(w);
                        expint = log((lam0+1)*u1/2+(1-lam0)/u1/2)/nu/b-a/2/b*dt;
                    }
                }
                else {
                    u0 = atan(lam0);
                    u1 = u0-nu*sqD/2*dt;
                    expint = log(cos(u1)/cos(u0))/nu/b-a/2/b*dt;
                }
            }
        }
    }

    for(i=0; i<nfluxes; i++){
        aij = aj_vector[i];
        a_check += aij;

        bij = bj_vector[i];
        b_check += bij;

        cij = cj_vector[i];
        c_check += cij;

        if(isnull(b) && isnull(c)){
            fluxes[i] += aij*dt-bij*e0/nu/a*(exp(-nu*a*dt)-1);
            fluxes[i] += cij/nu/a/e0*(exp(nu*a*dt)-1);
        } else {
            if(notnull(c)){
                A = aij-cij*a/c;
                B = bij-cij*b/c;
                C = cij/c;
                fluxes[i] += A*dt+B*expint+C*(s1-s0);
            } else {
                A = aij-bij*a/b;
                B = bij/b;
                C = cij-bij*c/b;
                fluxes[i] += A*dt+B*(s1-s0)+C*expint;
            }
        }
    }

    /* Check the coefficients sum to aoj, boj and coj */
    if(notnull(a-a_check)||notnull(b-b_check)||notnull(c-c_check))
        return REZEQ_ERROR + __LINE__;

    return 0;
}


/* Integrate reservoir equation over 1 time step and compute associated fluxes */
int c_integrate(int nalphas, int nfluxes,
                            double * alphas, double * scalings,
                            double nu,
                            double * a_matrix_noscaling,
                            double * b_matrix_noscaling,
                            double * c_matrix_noscaling,
                            double t0,
                            double s0,
                            double delta,
                            int *niter, double * s1, double * fluxes) {
    int REZEQ_DEBUG=0;

    int i, nit=0, jalpha_next;
    int outside_alpha_bounds=0;
    int is_below_alpha_min=0;
    double aoj=0., boj=0., coj=0.;
    double a=0, b=0, c=0;
    double funval=0, funval_prev=0;
    double alpha0, alpha1;
    double alpha_min=alphas[0];
    double alpha_max=alphas[nalphas-1];

    double a_vect[REZEQ_NFLUXES_MAX];
    double b_vect[REZEQ_NFLUXES_MAX];
    double c_vect[REZEQ_NFLUXES_MAX];

    /* Initial interval */
    int jmin = 0, jmax=nalphas-2;
    int jalpha = c_find_alpha(nalphas, alphas, s0);

    /* Initialise iteration */
    double aoj_prev=0., boj_prev=0., coj_prev=0.;

    /* Inialise other variables */
    int extrapolating_low = 0;
    int extrapolating_high = 0;
    double t_final = t0+delta;
    double t_start=t0, t_end=t0;
    double s_start=s0, s_interm=s0, s_end=s0;
    double continuity_error_max = 0;
    for(i=0; i<nfluxes; i++)
        fluxes[i] = 0;

    if(nfluxes>REZEQ_NFLUXES_MAX)
        return REZEQ_ERROR_NFLUXES_TOO_LARGE;

    if(REZEQ_DEBUG==1){
        fprintf(stdout, "\n\nStart integrate s0=%0.3f j=%d\n", s0, jalpha);
    }

    /* Time loop */
    while (ispos(t_final-t_end) && nit<nalphas) {
        nit += 1;

        /* Extrapolation is triggered if s_start is
         * below alpha0+EPS for low and
         * above alpha1-EPS for high */
        extrapolating_low = s_start<alpha_min;
        extrapolating_high = s_start>alpha_max;

        if(jalpha<0 || jalpha>nalphas-2)
            return REZEQ_ERROR_INTEGRATE_OUT_OF_BOUNDS;

        /* Get band limits */
        alpha0 = alphas[jalpha];
        alpha1 = alphas[jalpha+1];

        /* Store previous coefficients */
        aoj_prev = aoj;
        boj_prev = boj;
        coj_prev = coj;

        /* Sum coefficients accross fluxes */
        aoj = 0;
        boj = 0;
        coj = 0;
        for(i=0;i<nfluxes;i++){
            /* Multiply approximation coefficients by scaling */
            a = a_matrix_noscaling[nfluxes*jalpha+i]*scalings[i];
            b = b_matrix_noscaling[nfluxes*jalpha+i]*scalings[i];
            c = c_matrix_noscaling[nfluxes*jalpha+i]*scalings[i];

            /* if s is lower than alpha1 or higher than alpham
                then set approx_fun to constant
            */
            if(extrapolating_low){
                a = c_approx_fun(nu, a, b, c, alpha_min);
                b = 0;
                c = 0;
            }
            else if(extrapolating_high){
                a = c_approx_fun(nu, a, b, c, alpha_max);
                b = 0;
                c = 0;
            }
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
            return REZEQ_ERROR_INTEGRATE_NAN_COEFF;

        /* Get derivative at beginning of time step */
        funval = c_approx_fun(nu, aoj, boj, coj, s_start);

        /* Check continuity */
        if(nit>1){
            continuity_error_max = REZEQ_EPS*1e2+fabs(funval_prev)*1e-5;
            if(fabs(funval-funval_prev)>continuity_error_max) {
                if(REZEQ_DEBUG==1)
                    fprintf(stdout, "    jalpha=%d/%d funval(%0.5f, %0.5f, %0.5f, %0.5f, %0.5f)=%5.5e"
                                "   funvel_prev=%5.5e diff=%5.5e\n",
                                jalpha, nalphas-2, nu, aoj, boj, coj, s_start, funval, funval_prev,
                                            fabs(funval-funval_prev));
                return REZEQ_ERROR_INTEGRATE_NOT_CONTINUOUS;
            }
        }

       /* Check integration up to the next band limit */
        if(notnull(funval)){
            if(isneg(funval)){
                /* non-increasing function -> move to lower band if not extrapolating*/
                jalpha_next = extrapolating_high ? jalpha : jalpha>jmin ? jalpha-1 : jmin;

                if(extrapolating_high){
                    s_interm = alpha_max-2*REZEQ_EPS;
                } else {
                    s_interm = alpha0;
                    if(isequal(s_interm, s_start))
                        s_interm -= 2*REZEQ_EPS;
                }


            } else if (ispos(funval)){
                /* increasing function -> move to upper band if not extrapolating */
                jalpha_next = extrapolating_low ? jalpha : jalpha<jmax ? jalpha+1 : jmax;

                if(extrapolating_low) {
                    s_interm = alpha_min+2*REZEQ_EPS;
                } else {
                    s_interm = alpha1;
                    if(isequal(s_interm, s_start))
                        s_interm += 2*REZEQ_EPS;
                }

            }

            /* Compute time for which s(t) = s_end */
            t_end = t_start+c_integrate_inverse(nu, aoj, boj, coj, s_start, s_interm);

        } else {
            /* derivative is null -> finish iteration */
            jalpha_next = jalpha;
            t_end = t_final;
            s_end = s_start;
        }

        /* Set time to end of time step if finished iteration (t_end>t_final)
            or if t1 is nan (i.e. close to steady or never reaching t_final) */
        t_end = (t_end>t_final || isnan(t_end) || t_end<t_start) ? t_final : t_end;

        /* Recompute s_end - required only if finished iteration or
         * extrapolating. Skip if funval is null => steady */
        if(notnull(funval))
            s_end = c_integrate_forward(nu, aoj, boj, coj, t_start, s_start, t_end);

        if(REZEQ_DEBUG==1){
            fprintf(stdout, "  [%d] j=%d(%0.5f, %0.5f) -> %d : nu=%0.5f a=%0.5f b=%0.5f c=%0.5f f=%0.5f"
                                    " ex_l=%d  ex_h=%d\n",
                                        nit, jalpha, alpha0, alpha1, jalpha_next, nu, aoj, boj,
                                        coj, funval, extrapolating_low, extrapolating_high);
            fprintf(stdout, "        t=%0.5f->%0.5f/%0.5f  s=%4.4e->(%4.4e)->%4.4e\n\n",
                                        t_start, t_end, t_final, s_start, s_interm, s_end);
        }
        if(isnull(t_end-t_start)){
            return REZEQ_ERROR_INTEGRATE_TSTART_EQUAL_TEND;
        }

        /* Increment fluxes during the last interval */
        c_increment_fluxes(nfluxes, nu,
                    &(a_vect),
                    &(b_vect),
                    &(c_vect),
                    aoj, boj, coj, t_start, t_end, s_start, s_end, fluxes);

        /* Loop for next band */
        funval_prev = c_approx_fun(nu, aoj, boj, coj, s_end);
        t_start = t_end;
        s_start = s_end;
        jalpha = jalpha_next;
    }

    /* Store results */
    *s1 = s_end;
    *niter = nit;

    if(REZEQ_DEBUG==1){
        fprintf(stdout, "\nEnd integrate s1=%0.5f j=%d\n", s_end, jalpha);
    }

    /* Convergence problem */
    if(ispos(t_final-t_end)) {
        //fprintf(stdout, "\ndelta=%0.3f t0=%0.3f\n", delta, t0);
        return REZEQ_ERROR_INTEGRATE_NO_CONVERGENCE;
    }
    return 0;
}



#include "c_integ.h"

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
    double tmax, tmp=0, lam0=0, sqD=0;

    if(nu<0 || isnan(nu))
        return c_get_nan();

    if(isnull(b) && isnull(c)){
        /* Constant -> always valid */
        tmax = c_get_inf();
    }
    else if(isnull(a) && isnull(b) && notnull(c)){
        tmax = c<0 ? c_get_inf() : e0/nu/c;
    }
    else if(isnull(a) && notnull(b) && isnull(c)){
        tmax = b>0 ? c_get_inf() : -1/e0/nu/b;
    }
    else if(notnull(a) && isnull(b) && notnull(c)){
        tmax = c<0 || (c>0 && a<-c/e0) ? c_get_inf() : log(1+a*e0/c)/nu/a;
    }
    else if(notnull(a) && notnull(b) && isnull(c)){
        tmax = b>0 || (b<0 && a<b*e0) ? c_get_inf() : -log(1+a/e0/b)/nu/a;
    }
    else if(notnull(b) && notnull(c)){
        sqD = sqrt(fabs(Delta));
        lam0 = (2*b*e0+a)/sqD;

        if(isnull(Delta)){
            /* Determinant is zero */
            tmax = a<-2*e0*b ? -2/(a+2*b*e0)/nu : c_get_inf();
            tmax = fmin(tmax, a>-c/e0 ? 4*b*e0/(a+2*b*e0)/nu/a : c_get_inf());
        }
        else if (isneg(Delta)){
            tmax = lam0<0 ? atan(-1./lam0)*2/nu/sqD : c_get_inf();
            tmp = atan((lam0*sqD-a)/(a*lam0+sqD))*2/nu/sqD;
            tmp = tmp>0 ? tmp : c_get_inf();
            tmax = fmin(fmin(tmax, tmp), REZEQ_PI/nu/sqD);
        }
        else {
            tmax = lam0<-1 ? atanh(-1./lam0)*2/nu/sqD : c_get_inf();
            tmp = atanh((lam0*sqD-a)/(a*lam0-sqD))*2/nu/sqD;
            tmp = tmp>0 ? tmp : c_get_inf();
            tmax = fmin(tmax, tmp);
        }
    }
    return tmax>=0 ? tmax : c_get_nan();
}

/* Solution of dS/dt = f*(s) */
double c_integrate_forward(double nu, double a, double b, double c,
                        double t0, double s0, double t){
    double e0 = exp(-nu*s0);
    double sgn=1,omeg=0, lam0=0, sqD=0;
    double Delta = a*a-4*b*c;
    double ra2b, s1;

    if(nu<0 || isnan(nu))
        return c_get_nan();

    if(t<t0)
        return c_get_nan();

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
    double tau0=0, tau1=0, sqD=0., lam0=0., lam1=0.;

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
            return 1./sqD/nu*log((1+lam1)*(1-lam0)/(1-lam1)/(1+lam0));
        }
        else {
            return -2./sqD/nu*(atan(lam1)-atan(lam0));
        }
    }
    return c_get_nan();
}


/* Increment fluxes by integrating f*(s) */
int c_increment_fluxes(int nfluxes, double * scalings, double nu,
                        double * aj_vector_noscaling,
                        double * bj_vector_noscaling,
                        double * cj_vector_noscaling,
                        double aoj, double boj, double coj,
                        double t0, double t1, double s0, double s1,
                        double * fluxes){
    int i;
    double dt = t1-t0;
    double ds = s1-s0;
    double e0 = exp(-nu*s0);
    double expint=0;
    double a = aoj, a_check=0;
    double b = boj, b_check=0;
    double c = coj, c_check=0;
    double A, B, C;
    double Delta = aoj*aoj-4*boj*coj;
    double sqD, aij, bij, cij, gam, lam0, u0, u1;

    if(t1<t0 || nu<0 || isnan(nu))
        return REZEQ_ERROR + __LINE__;

    /* Integrate exp(-nuS) if needed */
    if(notnull(b) || notnull(c)){
        if(isnull(a) && isnull(b) && notnull(c)){
            expint = dt*e0-nu*c/2*(t1-t0)*(t1-t0);
        }
        else if(isnull(a) && notnull(b) && isnull(c)){
            expint = dt/e0+nu*b/2*(t1-t0)*(t1-t0);
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
                    u1 = exp(nu*sqD/2*dt);
                    expint = log((lam0+1)*u1/2+(1-lam0)/u1/2)/nu/b-a/2/b*dt;
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
        aij = aj_vector_noscaling[i]*scalings[i];
        a_check += aij;

        bij = bj_vector_noscaling[i]*scalings[i];
        b_check += bij;

        cij = cj_vector_noscaling[i]*scalings[i];
        c_check += cij;

        if(isnull(b) && isnull(c)){
            fluxes[i] += aij*dt-bij*e0/nu/a*(exp(-nu*a*(t1-t0))-1);
            fluxes[i] += cij/nu/a/e0*(exp(nu*a*(t1-t0))-1);
        } else {
            if(notnull(c)){
                A = aij-cij*a/c;
                B = bij-cij*b/c;
                C = cij/c;
                fluxes[i] += A*(t1-t0)+B*expint+C*(s1-s0);
            } else {
                A = aij-bij*a/b;
                B = bij/b;
                C = cij-bij*c/b;
                fluxes[i] += A*(t1-t0)+B*(s1-s0)+C*expint;
            }
        }
    }

    /* Check the coefficients sum to aoj, boj and coj */
    if(notnull(a-a_check)||notnull(b-b_check)||notnull(c-c_check))
        return REZEQ_ERROR + __LINE__;

    return 0;
}


/* Integrate reservoir equation over 1 time step and compute associated fluxes */
int c_integrate(int nalphas, int nfluxes, double delta,
                            double * alphas, double * scalings,
                            double * nu_vector,
                            double * a_matrix_noscaling,
                            double * b_matrix_noscaling,
                            double * c_matrix_noscaling,
                            double s0, double * s1, double * fluxes) {
    int i, jalpha_next;
    double aoj=0., boj=0., coj=0., nu;
    double a=0, b=0, c=0;
    double ds1=0, ds2=0;
    double alpha0, alpha1, t1;

    /* Initial interval */
    int jalpha = c_find_alpha(nalphas, alphas, s0);

    /* Initialise iteration */
    int is_low = s0<alphas[0];
    int is_high = s0>alphas[nalphas-1];
    double t0 = 0.;
    int niter = 0;
    double aoj_prev=0., boj_prev=0., coj_prev=0.;

    /* Inialise fluxes */
    for(i=0; i<nfluxes; i++)
        fluxes[i] = 0;

    /* Time loop */
    while (t0<delta-1e-10 && niter<nalphas) {
        niter += 1;

        /* Exponential factor */
        nu = nu_vector[jalpha];

        if(isnan(nu) || nu<0)
            return REZEQ_ERROR + __LINE__;

        /* Store previous coefficients */
        aoj_prev = aoj;
        boj_prev = boj;
        coj_prev = coj;

        /* Sum coefficients accross fluxes */
        aoj = 0;
        boj = 0;
        coj = 0;
        for(i=0;i<nfluxes;i++){
            a = a_matrix_noscaling[nfluxes*jalpha+i]*scalings[i];
            b = b_matrix_noscaling[nfluxes*jalpha+i]*scalings[i];
            c = c_matrix_noscaling[nfluxes*jalpha+i]*scalings[i];
            /* if s is lower than alpha1 or higher than alpham
                then set approx_fun to constant
            */
            if(is_low) {
                aoj += c_approx_fun(nu, a, b, c, alphas[0]);
            }
            else if(is_high) {
                aoj += c_approx_fun(nu, a, b, c, alphas[nalphas-1]);
            } else {
                aoj += a;
                boj += b;
                coj += c;
            }
        }

        if(isnan(aoj) || isnan(boj) || isnan(coj))
            return REZEQ_ERROR + __LINE__;

        /* Check continuity */
        if(niter>1){
            ds1 = c_approx_fun(nu, aoj_prev, boj_prev, coj_prev, s0);
            ds2 = c_approx_fun(nu, aoj, boj, coj, s0);
            if(notnull(ds1-ds2))
                return REZEQ_ERROR + __LINE__;
        }

        /* Get band limits */
        alpha0 = alphas[jalpha];
        alpha1 = alphas[jalpha+1];

        /* integrate ODE up to the end of the time step */
        *s1 = c_integrate_forward(t0, s0, nu, aoj, boj, coj, delta);

        /* divergent solution */
        if(isnan(*s1))
            return REZEQ_ERROR + __LINE__;

        /** Check if integration stays in the band or
        * if we are below lowest alphas or above highest alpha
        * In these cases, complete integration straight away.
        **/
        is_low = 0;
        is_high = 0;
        if(*s1>=alpha0 && *s1<=alpha1){
            c_increment_fluxes(nfluxes, scalings, nu,
                        &(a_matrix_noscaling[nfluxes*jalpha]),
                        &(b_matrix_noscaling[nfluxes*jalpha]),
                        &(c_matrix_noscaling[nfluxes*jalpha]),
                        aoj, boj, coj, t0, delta, s0, *s1, fluxes);
            t0 = delta;
            s0 = *s1;
        }
        else {
            is_low = *s1<alpha0;
            is_high = *s1>alpha1;
            if((jalpha==0 && is_low) || (jalpha==nalphas-2 && is_high)){
                /* We are on the fringe of the alphas domain */
                jalpha_next = jalpha;
                t1 = delta;
            }
            else {
                /* If not, decrease or increase parameter band
                 * depending on increasing or decreasing nature
                 * of ODE solution */
                if(*s1<=alpha0){
                    jalpha_next = jalpha-1;
                    *s1 = alpha0;
                } else{
                    jalpha_next = jalpha+1;
                    *s1 = alpha1;
                }

                /* Find time where we are crossing to the next band */
                t1 = t0+c_integrate_inverse(s0, nu, aoj, boj, coj, *s1);
            }
            /* Increment fluxes during the last interval */
            c_increment_fluxes(nfluxes, scalings,
                        nu_vector[jalpha],
                        &(a_matrix_noscaling[nfluxes*jalpha]),
                        &(b_matrix_noscaling[nfluxes*jalpha]),
                        &(c_matrix_noscaling[nfluxes*jalpha]),
                        aoj, boj, coj, t0, t1, s0, *s1, fluxes);
            t0 = t1;
            s0 = *s1;
            jalpha = jalpha_next;
        }
        //fprintf(stdout, "a=%0.2f b=%0.2f [j=%d] / t=%0.2f  u=%0.1f f=%0.1f\n", aoj, boj, jalpha, t1, *s1, fluxes[1]);

    }

    /* Convergence problem */
    if(notnull(t0-delta))
        return REZEQ_ERROR + __LINE__;

    return 0;
}



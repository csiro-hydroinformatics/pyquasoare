#include "c_integ.h"

double c_get_eps() {
    return REZEQ_EPS;
}

double c_get_nan() {
    static double zero=0.;
    double nan=zero/zero;
    return nan;
}

double c_get_inf() {
    static double zero=0.;
    double inf=1./zero;
    return inf;
}

double c_approx_fun(double nu, double a, double b, double c, double s){
    return a+b*exp(-nu*s)+c*exp(nu*s);
}

double c_approx_jac(double nu, double a, double b, double c, double s){
    return -nu*b*exp(-nu*s)+nu*c*exp(nu*s);
}


int isnull(double x){
    return fabs(x)<REZEQ_EPS ? 1 : 0;
}
int notnull(double x){
    return 1-isnull(x);
}

double c_steady_state(nu, a, b, c){

    return c_get_nan();
}

double c_integrate_delta_t_max(double nu, double a, double b, double c,
                                double s0){
    double e0 = exp(-nu*s0);
    double lam0=0, sqD=0;
    double Delta = a*a-4*b*c;
    double tmax, tmp=0;

    if(nu<0)
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
        else if (Delta<0){
            tmax = lam0<0 ? atan(-1./lam0)*2/nu/sqD : c_get_inf();
            tmp = atan((lam0*sqD-a)/(a*lam0+sqD))*2/nu/sqD;
            tmp = tmp>0 ? tmp : c_get_inf();
            tmax = fmin(fmin(tmax, tmp), REZEQ_PI/nu/sqD);
        }
        else {
            tmax = lam0<-1 ? atanh(-1./lam0)*2/nu/sqD : c_get_inf();
            tmp = atanh((lam0*sqD-a)/(a*lam0-sqD))*2/nu/sqD;
            tmp = tmp>0 ? tmp : c_get_inf();
            tmax = fmin(fmin(tmax, tmp), REZEQ_PI/nu/sqD);
        }
    }
    return tmax;
}


double c_integrate_forward(double nu, double a, double b, double c,
                        double t0, double s0, double t){
    double e0 = exp(-nu*s0);
    double sgn=1,omeg=0, lam0=0, sqD=0;
    double Delta = a*a-4*b*c;
    double rab = a/2/b;
    double s1;

    if(nu<0)
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
        if(isnull(Delta)){
            /* Determinant is zero */
            s1 = -log((e0+rab)/(1+(e0+rab)*nu*b*(t-t0))-rab)/nu;
        }
        else {
            /* Non zero determinant */
            sgn = Delta<0 ? -1 : 1;
            sqD = sqrt(sgn*Delta);
            omeg = Delta<0 ? tan(nu*sqD*(t-t0)/2) : tanh(nu*sqD*(t-t0)/2);
            lam0 = (2*b*e0+a)/sqD;
            s1 = -log((lam0+sgn*omeg)/(1+lam0*omeg)*sqD/2/b-rab)/nu;
        }
    }
    return s1;
}


double c_integrate_inverse(double nu, double a, double b, double c,
                                double s0, double s1){
    double e0 = exp(-nu*s0);
    double e1 = exp(-nu*s1);
    double sqD=0., lam0=0., lam1=0., omeginv1=0., omeginv0=0.;
    double Delta = a*a-4*b*c;
    double sgn = Delta<0 ? -1 : 1;
    double tau0=0., tau1=0.;

    if(nu<0)
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
        if(isnull(Delta)){
            /* Determinant is zero */
            tau0 = 2/(a+2*b*e0)/nu;
            tau1 = 2/(a+2*b*e1)/nu;
            return tau1-tau0;
        }
        else {
            /* Non zero determinant */
            sqD = sqrt(sgn*Delta);
            lam0 = (2*b*e0+a)/sqD;
            omeginv0 = Delta<0 ? atan(lam0) : atanh(lam0);
            omeginv1 = Delta<0 ? atan(lam1) : atanh(lam1);
            return 2*sgn/sqD*(omeginv1-omeginv0);
        }
    }

    return c_get_nan();
}


int c_find_alpha(int nalphas, double * alphas, double s0){
    int i=0;

    if(s0<=alphas[0])
        return 0;

    if(s0>alphas[nalphas-1])
        return nalphas-2;

    while(s0>alphas[i] && i<=nalphas-2){
        i++;
    }
    return i-1;
}

/* a and b matrix: [nalphas x nfluxes] */
int c_increment_fluxes(int nfluxes,
                        double * scalings,
                        double nu,
                        double * a_vector_noscaling,
                        double * b_vector_noscaling,
                        double * c_vector_noscaling,
                        double aoj,
                        double boj,
                        double coj,
                        double t0,
                        double t1,
                        double s0,
                        double s1,
                        double * fluxes){
    int i;
    double dt = t1-t0;
    double ds = s1-s0;
    double aij, bij, cij, dflux;

    if(t1<t0)
        return -1;

    for(i=0; i<nfluxes; i++){
        aij = a_vector_noscaling[i]*scalings[i];
        bij = b_vector_noscaling[i]*scalings[i];
        cij = c_vector_noscaling[i]*scalings[i];

        /* TODO */
        if(fabs(boj)>REZEQ_EPS){
            dflux = -999;
        } else {
            /* u = s0+ai(t-t0) */
            dflux = -999;
        }
        fluxes[i] += dflux;
    }

    return 0;
}


/**
* Integrate reservoir equation over 1 time step:
* - scalings [nfluxes] : scalings applied to linear coefficients
* - a and b matrices [nalphas x nfluxes] : reservoir function piecewise linear
*                                           approx
* - s0 : initial state
* - s1 : final state
**/
int c_integrate(int nalphas, int nfluxes, double delta,
                            double * alphas,
                            double * scalings,
                            double * nu_vector,
                            double * a_matrix_noscaling,
                            double * b_matrix_noscaling,
                            double * c_matrix_noscaling,
                            double s0,
                            double * s1,
                            double * fluxes) {
    int i, jalpha_next;
    double aoj=0., boj=0., coj=0., nu;
    double ds1=0, ds2=0;
    double alow, ahigh, t1;

    /* Initial interval */
    int jalpha = c_find_alpha(nalphas, alphas, s0);

    /* Initialise iteration */
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
            aoj += a_matrix_noscaling[nfluxes*jalpha+i]*scalings[i];
            boj += b_matrix_noscaling[nfluxes*jalpha+i]*scalings[i];
            coj += c_matrix_noscaling[nfluxes*jalpha+i]*scalings[i];
        }

        if(isnan(aoj))
            return REZEQ_ERROR + __LINE__;

        if(isnan(boj))
            return REZEQ_ERROR + __LINE__;

        if(isnan(coj))
            return REZEQ_ERROR + __LINE__;

        /* Check continuity */
        if(niter>1){
            ds1 = c_approx_fun(nu, aoj_prev, boj_prev, coj_prev, s0);
            ds2 = c_approx_fun(nu, aoj, boj, coj, s0);
            if(fabs(ds1-ds2)>REZEQ_EPS)
                return REZEQ_ERROR + __LINE__;
        }

        /* Get band limits */
        alow = alphas[jalpha];
        ahigh = alphas[jalpha+1];

        /* integrate ODE up to the end of the time step */
        *s1 = c_integrate_forward(t0, s0, nu, aoj, boj, coj, delta);

        /** Check if integration stays in the band or
        * if we are below lowest alphas or above highest alpha
        * In these cases, complete integration straight away.
        **/
        if(*s1>=alow && *s1<=ahigh){
            c_increment_fluxes(nfluxes, scalings, nu,
                        &(a_matrix_noscaling[nfluxes*jalpha]),
                        &(b_matrix_noscaling[nfluxes*jalpha]),
                        &(c_matrix_noscaling[nfluxes*jalpha]),
                        aoj, boj, coj, t0, delta, s0, *s1, fluxes);
            t0 = delta;
            s0 = *s1;
        }
        else {
            if((jalpha==0 && *s1<alow) || (jalpha==nalphas-2 && *s1>ahigh)){
                /* We are on the fringe of the alphas domain */
                jalpha_next = jalpha;
                t1 = delta;
            }
            else {
                /* If not, decrease or increase parameter band
                 * depending on increasing or decreasing nature
                 * of ODE solution */
                if(*s1<=alow){
                    jalpha_next = jalpha-1;
                    *s1 = alow;
                } else{
                    jalpha_next = jalpha+1;
                    *s1 = ahigh;
                }

                /* Find time where we move to the next band */
                t1 = t0+c_integrate_inverse(s0, nu, aoj, boj, coj, *s1);
            }
            /* Increment variables */
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
    if(t0<delta-1e-10)
        return REZEQ_ERROR + __LINE__;

    return 0;
}


/**
* Integrate reservoir equation over multiple time steps:
* - scalings [nval, nfluxes] : scalings applied to linear coefficients
* - other input args identical to c_integrate
* - s1 [nval] : final states
* - fluxes [nval, nfluxes] : flux computed
**/
int c_run(int nalphas, int nfluxes, int nval, double delta,
                            double * alphas,
                            double * scalings,
                            double * nu_vector,
                            double * a_matrix_noscaling,
                            double * b_matrix_noscaling,
                            double * c_matrix_noscaling,
                            double s0,
                            double * s1,
                            double * fluxes) {
    int ierr, t;

    for(t=0; t<nval; t++){
        ierr = c_integrate(nalphas, nfluxes, delta,
                            alphas,
                            &(scalings[nfluxes*t]),
                            nu_vector,
                            a_matrix_noscaling,
                            b_matrix_noscaling,
                            c_matrix_noscaling,
                            s0,
                            &(s1[t]),
                            &(fluxes[nfluxes*t]));
        if(ierr>0)
            return ierr;

        /* Loop initial state */
        s0 = s1[t];
    }

    return 0;
}

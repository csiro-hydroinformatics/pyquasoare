#include "c_integ.h"

double c_integrate_forward(double t0, double u0, double a, double b, double t){
    if(fabs(b)>REZEQ_EPS){
        return -a/b+(u0+a/b)*exp(b*(t-t0));
    }
    else{
        return u0+a*(t-t0);
    }
}

double c_integrate_inverse(double t0, double u0, double a, double b, double u){
    if(fabs(b)>REZEQ_EPS){
        return t0+log((a+b*u)/(a+b*u0))/b;
    }
    else{
        return t0+(u-u0)/a;
    }
}

int c_find_alpha(int nalphas, double * alphas, double u0){
    int i=0;

    if(u0<=alphas[0])
        return 0;

    if(u0>alphas[nalphas-1])
        return nalphas-2;

    while(u0>alphas[i] && i<=nalphas-2){
        i++;
    }
    return i-1;
}

/* a and b matrix: [nalphas x nfluxes] */
int c_increment_fluxes(int nalphas, int nfluxes,
                        double * scalings,
                        double * a_matrix_noscaling,
                        double * b_matrix_noscaling,
                        int jalpha,
                        double aoj,
                        double boj,
                        double t0,
                        double t1,
                        double u0,
                        double u1,
                        double * fluxes){
    int j;
    double dt = t1-t0;
    double du = u1-u0;
    double aij, bij, dflux;

    if(jalpha<0 || jalpha>nalphas-2)
        return -1;

    if(t1<t0)
        return -1;

    for(j=0; j<nfluxes; j++){
        aij = a_matrix_noscaling[nfluxes*jalpha+j]*scalings[j];
        bij = b_matrix_noscaling[nfluxes*jalpha+j]*scalings[j];

        if(fabs(boj)>REZEQ_EPS){
            dflux = bij/boj*du+(aij-aoj*bij/boj)*dt;
        } else {
            /* u = u0+ai(t-t0) */
            dflux = (aij+bij*u0)*dt+bij*aoj*dt*dt/2;
        }
        fluxes[j] += dflux;
    }

    return 0;
}


/**
* Integrate reservoir equation over 1 time step:
* - scalings [nfluxes] : scalings applied to linear coefficients
* - a and b matrices [nalphas x nfluxes] : reservoir function piecewise linear
*                                           approx
* - u0 : initial state
* - u1 : final state
**/
int c_integrate(int nalphas, int nfluxes, double delta,
                            double * alphas,
                            double * scalings,
                            double * a_matrix_noscaling,
                            double * b_matrix_noscaling,
                            double u0,
                            double * u1,
                            double * fluxes) {
    int j, jalpha_next;
    double aoj=0., boj=0.;
    double du1=0, du2=0;
    double ulow, uhigh, t1;

    /* Initial interval */
    int jalpha = c_find_alpha(nalphas, alphas, u0);

    /* Initialise iteration */
    double t0 = 0.;
    int niter = 0;
    double aoj_prev=0., boj_prev=0.;

    /* Inialise fluxes */
    for(j=0; j<nfluxes; j++)
        fluxes[j] = 0;

    /* Time loop */
    while (t0<delta-1e-10 && niter<nalphas) {
        niter += 1;

        /* Store previous coefficients */
        aoj_prev = aoj;
        boj_prev = boj;

        /* Sum coefficients accross fluxes */
        aoj = 0;
        boj = 0;
        for(j=0;j<nfluxes;j++){
            aoj += a_matrix_noscaling[nfluxes*jalpha+j]*scalings[j];
            boj += b_matrix_noscaling[nfluxes*jalpha+j]*scalings[j];
        }

        if(isnan(aoj))
            return REZEQ_ERROR + __LINE__;

        if(isnan(boj))
            return REZEQ_ERROR + __LINE__;

        /* Check continuity */
        if(niter>1){
            du1 = aoj_prev+boj_prev*u0;
            du2 = aoj+boj*u0;
            if(fabs(du1-du2)>1e-10)
                return REZEQ_ERROR + __LINE__;
        }

        /* Get band limits */
        ulow = alphas[jalpha];
        uhigh = alphas[jalpha+1];

        /* integrate ODE up to the end of the time step */
        *u1 = c_integrate_forward(t0, u0, aoj, boj, delta);

        /** Check if integration stays in the band or
        * if we are below lowest alphas or above highest alpha
        * In these cases, complete integration straight away.
        **/
        if(*u1>=ulow && *u1<=uhigh){
            c_increment_fluxes(nalphas, nfluxes, scalings,
                        a_matrix_noscaling, b_matrix_noscaling,
                        jalpha, aoj, boj, t0, delta, u0, *u1, fluxes);
            t0 = delta;
            u0 = *u1;
        }
        else {
            if((jalpha==0 && *u1<ulow) || (jalpha==nalphas-2 && *u1>uhigh)){
                /* We are on the fringe of the alphas domain */
                jalpha_next = jalpha;
                t1 = delta;
            }
            else {
                /* If not, decrease or increase parameter band
                 * depending on increasing or decreasing nature
                 * of ODE solution */
                if(*u1<=ulow){
                    jalpha_next = jalpha-1;
                    *u1 = ulow;
                } else{
                    jalpha_next = jalpha+1;
                    *u1 = uhigh;
                }

                /* Find time where we move to the next band */
                t1 = c_integrate_inverse(t0, u0, aoj, boj, *u1);
            }
            /* Increment variables */
            c_increment_fluxes(nalphas, nfluxes, scalings,
                        a_matrix_noscaling, b_matrix_noscaling,
                        jalpha, aoj, boj, t0, t1, u0, *u1, fluxes);
            t0 = t1;
            u0 = *u1;
            jalpha = jalpha_next;
        }
        //fprintf(stdout, "a=%0.2f b=%0.2f [j=%d] / t=%0.2f  u=%0.1f f=%0.1f\n", aoj, boj, jalpha, t1, *u1, fluxes[1]);

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
* - u1 [nval] : final states
* - fluxes [nval, nfluxes] : flux computed
**/
int c_run(int nalphas, int nfluxes, int nval, double delta,
                            double * alphas,
                            double * scalings,
                            double * a_matrix_noscaling,
                            double * b_matrix_noscaling,
                            double u0,
                            double * u1,
                            double * fluxes) {
    int ierr, t;

    for(t=0; t<nval; t++){
        ierr = c_integrate(nalphas, nfluxes, delta,
                            alphas,
                            &(scalings[nfluxes*t]),
                            a_matrix_noscaling,
                            b_matrix_noscaling,
                            u0,
                            &(u1[t]),
                            &(fluxes[nfluxes*t]));
        if(ierr>0)
            return ierr;

        /* Loop initial state */
        u0 = u1[t];
    }

    return 0;
}

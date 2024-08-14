#include "c_rezeq_quad.h"

/* Approximation functions */
double c_quad_fun(double a, double b, double c, double s){
    return a*s*s+b*s+c;
}

double c_quad_grad(double a, double b, double c, double s){
    return 2*a*s+b;
}

/* Code pasted from
 * https://stackoverflow.com/questions/48979861/numerically-stable-method-for-solving-quadratic-equations
 */

double diff_of_products(double a, double b, double c, double d)
{
    double w = d*c;
    double e = fma(-d, c, w);
    double f = fma(a, b, -w);
    return f+e;
}

int c_quad_steady(double a, double b, double c, double steady[2]){
    double Delta = b*b-4*a*c;
    double q;

    steady[0] = c_get_nan();
    steady[1] = c_get_nan();

    if(notnull(a)){
        if(Delta>=0) {
            q = -0.5*(b+copysign(sqrt(diff_of_products(b, b, 4.0*a, c)), b));
            steady[0] = q/a;
            steady[1] = c/q;
        }
    }
    else {
        if(notnull(b)){
            steady[0] = -c/b;
        }
    }
    //if (isnull(a) && notnull(b)) {
    //    /* Linear function */
    //    steady[0] = -c/b;
    //}
    //else {
    //    if(Delta>=0 && Delta<REZEQ_EPS) {
    //        steady[0] = -b/2/a;
    //    }
    //    else if(Delta>REZEQ_EPS) {
    //        r1 = (-b-qD)/2/a;
    //        r2 = (-b+qD)/2/a;
    //        steady[0] = a>0 ? r1 : r2;
    //        steady[1] = a>0 ? r2 : r1;
    //    }
    //}
    //

    return 1;
}

/*
 * Quadratic interpolation coefficients to match a function such that
 * f(a0) = f0 , f(a1) = f1 , f((a0+a1)/2) = fm
 * fapprox(s) = as^2+bs+c with coefs = [a, b, c]
 * */
int c_quad_coefficients(double a0, double a1,
                            double f0, double f1, double fm,
                            double coefs[3]){
     double da = a1-a0;
     double A=0, B=0, C=0;

     /* Ensures quadratic function remains monotone */
     double f25=(3*f0+f1)/4, f75=(f0+3*f1)/4;
     double bnd1 = f25<f75 ? f25 : f75;
     double bnd2 = f25<f75 ? f75 : f25;
     fm = fm<bnd1 ? bnd1 : fm>bnd2 ? bnd2 : fm;

     if(isnull(da)){
        /* a0=a1, cannot interpolate */
        coefs[0] = c_get_nan();
        coefs[1] = c_get_nan();
        coefs[2] = c_get_nan();
        return REZEQ_QUAD_APPROX_SAMEALPHA;
     }
     else {
        /* Reparam s -> x
         * s = a0 + (a1-a0)x / x = (s-a0)/(a1-a0)
         * x^2 = (s^2-2a0s+a0^2)/(a1-a0)^2
         *
         * Approx functions
         * f0 = (2x-1)(x-1) = 2x^2-3x+1
         * f1 = (2x-1)x = 2x^2-x
         * fm = 4x(1-x) = 4x-4x^2
         *
         * f(x) = f0(2x^2-3x+1)+f1(2x^2-x)+fm(4x-4x^2)
         *      = x^2 (2f0+2f1-4fm) + x (4fm-f1-3f0) + f0
         *      = (s^2-2a0.s+a0^2)/da^2 A + (s-a0)/da B + C
         *      = A/da^2 s^2 + (-2a0.A/da^2+B/da) s +
         */
        A = 2*f0+2*f1-4*fm;
        B = 4*fm-f1-3*f0;
        C = f0;
        coefs[0] = A/da/da;
        coefs[1] = -2*a0*A/da/da+B/da;
        coefs[2] = A*a0*a0/da/da-B*a0/da+C;
     }
     return 0;
}

/* solution valididty range */
double c_quad_delta_t_max(double a, double b, double c, double s0){
    double Delta = b*b-4*a*c;
    double qD = sqrt(fabs(Delta));
    double delta_tmax=0;

    if(isnull(a)){
        delta_tmax = c_get_inf();
    }
    else{
        if(isnull(Delta)){
            delta_tmax = 1./(a*s0+b/2);
            delta_tmax = isnan(delta_tmax) || delta_tmax<0 ? c_get_inf() : delta_tmax;
        }
        else if (Delta>0){
            delta_tmax = c_get_inf();
        }
        else {
            delta_tmax = c_get_inf();
        }
    }

   return delta_tmax;
}

/* Solution of dS/dt = f*(s) */
double c_quad_forward(double a, double b, double c,
                        double t0, double s0, double t){
    double s1=0, omega = 0;

    double Delta = b*b-4*a*c;
    double qD = sqrt(fabs(Delta));
    double dt=t-t0;
    double delta_tmax = c_quad_delta_t_max(a, b, c, s0);

    if(t<t0)
        return c_get_nan();

    if(isequal(t, t0, REZEQ_EPS, 0.))
        return s0;

    if(t>t0+delta_tmax)
        return c_get_nan();

    if(isnull(a) && isnull(b)){
        s1 = s0+c*dt;
    }
    else if(isnull(a) && notnull(b)){
        s1 = -c/b+(s0+c/b)*exp(b*dt);
    }
    else{
        s1 = -b/2/a;
        if(isnull(Delta)){
            s1 += (s0+b/2/a)/(1-a*dt*b/2/a);
        }
        else if (Delta>0){
            omega = tanh(qD*dt/2);
            s1 += (s0+b/2/a-qD/2/a*omega)/(1-(2*a*s0+b)/qD*omega);
        }
        else {
            omega = tan(qD*dt/2);
            s1 += (2*a*s0+b+qD*omega)/(qD-(2*a*s0+b)*omega);
        }
    }
    return s1;
}

/* Primitive of 1/f*(s) */
double c_quad_inverse(double a, double b, double c,
                                double s0, double s1){
    double Delta = b*b-4*a*c;
    double qD = sqrt(fabs(Delta));

    if(isnull(a) && isnull(b)){
        return (s1-s0)/c;
    }
    else if(isnull(a) && notnull(b)){
        return 1/b*log(fabs((b*s1+c)/(b*s0+c)));
    }
    else{
        if(isnull(Delta)){
            return 2/(2*a*s0+b)-2/(2*a*s1+b);
        }
        else if (Delta>0){
            return -2/qD*atanh((2*a*s1+b)/qD)+2/qD*atanh((2*a*s1+b)/qD);
        }
        else {
            return 2/qD*atan((2*a*s1+b)/qD)-2/qD*atan((2*a*s1+b)/qD);
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
                        double t0, double t1, double s0, double s1,
                        double * fluxes){
    int i;
    double dt = t1-t0;
    double aij, bij, cij;
    double a = aoj, a_check=0;
    double b = boj, b_check=0;
    double c = coj, c_check=0;
    double Delta = boj*boj-4*aoj*coj;

    if(t1<t0)
        return REZEQ_QUAD_TIME_TOOLOW;

    for(i=0; i<nfluxes; i++){
        aij = aj_vector[i];
        a_check += aij;

        bij = bj_vector[i];
        b_check += bij;

        cij = cj_vector[i];
        c_check += cij;

        if(isnull(a) && isnull(b)){
        }
        else if(isnull(a) && notnull(b)){
        }
        else {
        }
    }

    if(notnull(a-a_check)||notnull(b-b_check)||notnull(c-c_check))
        return REZEQ_QUAD_FAILEDSUMCHECK;

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
                            double delta,
                            int *niter, double * s1, double * fluxes) {
    int REZEQ_DEBUG=0;

    int i, nit=0, jalpha_next=0;
    double aoj=0., boj=0., coj=0.;
    double a=0, b=0, c=0;
    double funval=0, funval_prev=0, grad=0;
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
    int jmin = 0, jmax=nalphas-2;
    int jalpha = c_find_alpha(nalphas, alphas, s0);

    /* Inialise variables */
    int extrapolating = 0;
    int extrapolating_low = 0;
    int extrapolating_high = 0;
    double t_final = t0+delta;
    double t_start=t0, t_end=t0;
    double s_start=s0, s_end=s0;
    for(i=0; i<nfluxes; i++)
        fluxes[i] = 0;

    if(nfluxes>REZEQ_NFLUXES_MAX)
        return REZEQ_QUAD_NFLUXES_TOO_LARGE;

    if(REZEQ_DEBUG==1){
        fprintf(stdout, "\n\nStart integrate s0=%0.3f j=%d\n", s0, jalpha);
    }

    /* Time loop */
    while ((t_final-t_end>0) && nit<niter_max) {
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
        aoj = 0; boj = 0; coj = 0;

        for(i=0;i<nfluxes;i++){
            /* Multiply approximation coefficients by scaling.
               if s is lower than alpha1 or higher than alpham
                then set approx_fun to linear extrapolation
                matching gradient at boundary.
            */
            if(extrapolating_low){
                a = a_matrix_noscaling[i]*scalings[i];
                b = b_matrix_noscaling[i]*scalings[i];
                c = c_matrix_noscaling[i]*scalings[i];

                grad = c_quad_grad(a, b, c, alpha_min);
                c = c_quad_fun(a, b, c, alpha_min)-grad*alpha_min;
                b = grad;
                a = 0;
            }
            else if(extrapolating_high){
                a = a_matrix_noscaling[nfluxes*(nalphas-2)+i]*scalings[i];
                b = b_matrix_noscaling[nfluxes*(nalphas-2)+i]*scalings[i];
                c = c_matrix_noscaling[nfluxes*(nalphas-2)+i]*scalings[i];

                grad = c_quad_grad(a, b, c, alpha_max);
                c = c_quad_fun(a, b, c, alpha_max)-grad*alpha_max;
                b = grad;
                a = 0;
            }
            else {
                a = a_matrix_noscaling[nfluxes*jalpha+i]*scalings[i];
                b = b_matrix_noscaling[nfluxes*jalpha+i]*scalings[i];
                c = c_matrix_noscaling[nfluxes*jalpha+i]*scalings[i];
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
            return REZEQ_QUAD_NAN_COEFF;

        /* Get derivative at beginning of time step */
        funval = c_quad_fun(aoj, boj, coj, s_start);

        /* Check continuity */
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
        s_end = c_quad_forward(aoj, boj, coj, t_start, s_start, t_final);

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
                /* Extrapolation, we cut integration at t_final */
                t_end = t_final;
                /* we also need to correct s_end because it is infinite */
                s_end = c_quad_forward(aoj, boj, coj, t_start, s_start, t_end);
                /* Ensure that s_end remains inside interpolation range */
                if(funval<0){
                    s_end = s_end < alpha_max-2*REZEQ_EPS ?
                                        alpha_max-2*REZEQ_EPS : s_end;
                }
                else{
                    s_end = s_end > alpha_min+2*REZEQ_EPS ?
                                        alpha_min+2*REZEQ_EPS : s_end;
                }
            }
            else {
                /* No extrapolation, we can reach s_end */
                t_end = t_start+c_quad_inverse(aoj, boj, coj, s_start, s_end);
            }
        }

        if(REZEQ_DEBUG==1){
            fprintf(stdout, "\t[%d] ex_l=%d  ex_h=%d\n",
                            nit, extrapolating_low, extrapolating_high);
            fprintf(stdout, "\t     j=%d (%2.2e, %2.2e) -> %d\n",
                                    jalpha, alpha0, alpha1, jalpha_next);
            fprintf(stdout, "\t     a=%4.4e  b=%4.4e  c=%4.4e  f=%4.4e\n",
                                    aoj, boj, coj, funval);
            fprintf(stdout, "\t     t=%4.4e-> %4.4e / %4.4e\n",
                                    t_start, t_end, t_final);
            fprintf(stdout, "\t     s=%4.4e -> %4.4e\n", s_start, s_end);
        }
        //if(isnull(t_end-t_start)){
        //    return REZEQ_ERROR_INTEGRATE_TSTART_EQUAL_TEND;
        //}

        /* Increment fluxes during the last interval */
        c_quad_fluxes(nfluxes,
                    &(a_vect), &(b_vect), &(c_vect),
                    aoj, boj, coj,
                    t_start, t_end, s_start, s_end, fluxes);

        /* Loop for next band */
        funval_prev = c_quad_fun(aoj, boj, coj, s_end);
        if(REZEQ_DEBUG==1){
            fprintf(stdout, "\t     funval(%0.5f, %0.5f, %0.5f, %0.5f)=%5.5e\n\n",
                                    aoj, boj, coj, s_end, funval_prev);
        }
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
int c_quad_model(int nalphas, int nfluxes, int nval, double delta,
                            double * alphas, double * scalings,
                            double * a_matrix_noscaling,
                            double * b_matrix_noscaling,
                            double * c_matrix_noscaling,
                            double s0, int * niter,
                            double * s1, double * fluxes) {
    int ierr, t;
    double t0=0;

    for(t=0; t<nval; t++){
        ierr = c_quad_integrate(nalphas, nfluxes, alphas,
                            &(scalings[nfluxes*t]),
                            a_matrix_noscaling,
                            b_matrix_noscaling,
                            c_matrix_noscaling,
                            t0, s0, delta,
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

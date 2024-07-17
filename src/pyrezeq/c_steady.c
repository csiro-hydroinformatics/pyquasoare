#include "c_steady.h"

int c_steady_state(double nu, double a, double b, double c, double steady[2]){
    double Delta = a*a-4*b*c;
    double sqD, p, q;
    double s1 = c_get_nan();
    double s2 = c_get_nan();
    steady[0] = s1;
    steady[1] = s2;

    /* Cannot have a steady state when
        - all params are pos
        - b and c are zeros
    */
    if((ispos(a) && ispos(b) && ispos(c))
                || (isnan(a)||isnan(b)||isnan(c))
                || (isnull(b)&&isnull(c)) ) {
        return 0;
    }

    if(notnull(b) && isnull(c)){
        steady[0] = notnull(a) ? -log(-a/b)/nu : s1;
    }
    else if(isnull(b) && notnull(c)){
        steady[0] = notnull(a) ? log(-a/c)/nu : s1;
    }
    else if(notnull(b) && notnull(c)){
        if(isnull(Delta)){
            steady[0] = notnull(a) ? log(-a/2/c)/nu : s1;
        }
        else if(ispos(Delta)){
            p = a/c;
            q = b/c;
            sqD = sqrt(p*p-4*q);
            s1 = log((-p-sqD)/2)/nu;
            s2 = log((-p+sqD)/2)/nu;
            steady[0] = isnan(s1) ? s2 : s1;
            steady[1] = isnan(s1) ? c_get_nan() : s2;
        }
    }
    return 0;
}


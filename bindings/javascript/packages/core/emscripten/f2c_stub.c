// Stub for f2c MAIN__ symbol required by libf2c.a when linking without Fortran main.
int MAIN__(void) {
    return 0;
}

#ifdef QSIM_HAS_CLAPACK
#ifndef NO_OVERWRITE
#include <stdlib.h>
#include <string.h>
#endif
#include "f2c.h"
#undef abs
#ifndef NO_OVERWRITE
extern char *F77_aloc(ftnlen, const char *);
#endif

int s_copy(char *a, char *b, ftnlen la, ftnlen lb) {
    char *aend = a + la;

    if (la <= lb) {
#ifndef NO_OVERWRITE
        if (a <= b || a >= b + la)
#endif
            while (a < aend) *a++ = *b++;
#ifndef NO_OVERWRITE
        else
            for (b += la; a < aend;) *--aend = *--b;
#endif
    } else {
        char *bend = b + lb;
#ifndef NO_OVERWRITE
        if (a <= b || a >= bend)
#endif
            while (b < bend) *a++ = *b++;
#ifndef NO_OVERWRITE
        else {
            a += lb;
            while (b < bend) *--a = *--bend;
            a += lb;
        }
#endif
        while (a < aend) *a++ = ' ';
    }
    return 0;
}

int s_cat(char *lp, char *rpp[], ftnint rnp[], ftnint *np, ftnlen ll) {
    ftnlen i, nc;
    char *rp;
    ftnlen n = *np;
#ifndef NO_OVERWRITE
    ftnlen L, m;
    char *lp0, *lp1;

    lp0 = 0;
    lp1 = lp;
    L = ll;
    i = 0;
    while (i < n) {
        rp = rpp[i];
        m = rnp[i++];
        if (rp >= lp1 || rp + m <= lp) {
            if ((L -= m) <= 0) {
                n = i;
                break;
            }
            lp1 += m;
            continue;
        }
        lp0 = lp;
        lp = lp1 = F77_aloc(L = ll, "s_cat");
        break;
    }
    lp1 = lp;
#endif
    for (i = 0; i < n; ++i) {
        nc = ll;
        if (rnp[i] < nc) nc = rnp[i];
        ll -= nc;
        rp = rpp[i];
        while (--nc >= 0) *lp++ = *rp++;
    }
    while (--ll >= 0) *lp++ = ' ';
#ifndef NO_OVERWRITE
    if (lp0) {
        memcpy(lp0, lp1, L);
        free(lp1);
    }
#endif
    return 0;
}
#endif

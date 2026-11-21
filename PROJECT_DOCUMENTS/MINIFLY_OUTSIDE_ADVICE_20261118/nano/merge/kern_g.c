
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#define NL 40
#define KMAX 400
#define SMAX 65
static inline int level(float n, float u) {
    int bn = 0; int64_t m = (int64_t)n; while (m > 1 && bn < 9) { m >>= 1; bn++; }
    return bn * 4 + (u <= 1 ? 0 : u <= 2 ? 1 : u <= 4 ? 2 : 3);
}
/* at each byte: every active cell that has counted says s(v) = log((c(v)+e)/(n+S e)); the answering cell's branches compete on the sum of trust times s;
   then each trust that was used changes by its step times (mismatch . s), and the cells count (nano/table/trust.py, with S a parameter) */
void mix_steps(int64_t n, int K1, int S, const int32_t *ids, const int32_t *y, float *C, float *N, float *U, double *T, double *Tu,
               double eta, double e, int learn, int shrink, double *P) {
    static double lt[4096], s[KMAX][SMAX]; double a[SMAX], p[SMAX]; static int lev[KMAX];
    for (int c = 0; c < 4096; c++) lt[c] = log(c + e);
    for (int64_t t = 0; t < n; t++) {
        const int32_t *id = ids + t * K1; double mx = -1e300, z = 0;
        for (int v = 0; v < S; v++) a[v] = 0;
        for (int k = 0; k < K1; k++) {
            lev[k] = -1; if (id[k] < 0 || N[id[k]] < 1) continue;
            const float *c = C + (int64_t)id[k] * S; lev[k] = level(N[id[k]], U[id[k]]); const double tk = T[k * NL + lev[k]], den = log(N[id[k]] + S * e);
            for (int v = 0; v < S; v++) { const double sv = (c[v] < 4096 ? lt[(int)c[v]] : log(c[v] + e)) - den; s[k][v] = sv; a[v] += tk * sv; }
        }
        for (int v = 0; v < S; v++) if (a[v] > mx) mx = a[v];
        for (int v = 0; v < S; v++) { p[v] = exp(a[v] - mx); z += p[v]; }
        for (int v = 0; v < S; v++) p[v] /= z;
        if (P) for (int v = 0; v < S; v++) P[t * S + v] = p[v];
        if (learn) {
            for (int k = 0; k < K1; k++) if (lev[k] >= 0) {
                double g = s[k][y[t]]; for (int v = 0; v < S; v++) g -= p[v] * s[k][v];
                T[k * NL + lev[k]] += (shrink ? eta / sqrt(1.0 + Tu[k * NL + lev[k]]) : eta) * g; Tu[k * NL + lev[k]] += 1.0;
            }
            for (int k = 0; k < K1; k++) { const int64_t i = id[k] < -1 ? -2 - (int64_t)id[k] : id[k];      /* an address below -1: the cell -2-address counts and is silent */
                if (i >= 0) { float *c = C + i * S; if (c[y[t]] == 0) U[i] += 1; c[y[t]] += 1; N[i] += 1; } }
        }
    }
}

/* The search for classes of one pair (X, Z). R-cells i = 0..m-1 with x index xc[i] and z index zc[i]; counts sparse (cptr, sym, cnt), totals n.
   J = sum over R-cells and bytes of c(v) log((C(v) - c(v) + e) / (N - n + S e)), C and N the pooled counts of the cell's class at its z. */
typedef struct { int m, nx, S; const int32_t *xc, *zc, *zptr, *zcell, *xptr, *xcell, *cptr, *sym; const float *cnt, *n; double e;
                 double *Cz, *Nz, *bef, *aft, *dl, *ci; int32_t *tz, *td; char *seen, *ind; int ntd; } W;
static void delta_all(W *w, int x, const int32_t *cls) {
    const int S = w->S; const double e = w->e, lS = log((double)S); w->ntd = 0;
    for (int a = w->xptr[x]; a < w->xptr[x + 1]; a++) {
        const int i = w->xcell[a], z = w->zc[i]; int ntz = 0;
        if (w->zptr[z + 1] - w->zptr[z] < 2) continue;
        for (int q = w->cptr[i]; q < w->cptr[i + 1]; q++) w->ci[w->sym[q]] = w->cnt[q];
        for (int b = w->zptr[z]; b < w->zptr[z + 1]; b++) {
            const int j = w->zcell[b]; if (j == i) continue; const int c = cls[w->xc[j]]; if (c < 0) continue;
            if (!w->seen[c]) { w->seen[c] = 1; w->tz[ntz++] = c; w->Nz[c] = 0; w->bef[c] = 0; w->aft[c] = 0; }
            w->Nz[c] += w->n[j]; for (int q = w->cptr[j]; q < w->cptr[j + 1]; q++) w->Cz[(int64_t)c * S + w->sym[q]] += w->cnt[q];
        }
        for (int b = w->zptr[z]; b < w->zptr[z + 1]; b++) {
            const int j = w->zcell[b]; if (j == i) continue; const int c = cls[w->xc[j]]; if (c < 0) continue;
            const double lb = log(w->Nz[c] - w->n[j] + S * e), la = log(w->Nz[c] + w->n[i] - w->n[j] + S * e); const double *Cc = w->Cz + (int64_t)c * S;
            for (int q = w->cptr[j]; q < w->cptr[j + 1]; q++) { const int v = w->sym[q]; const double cv = w->cnt[q];
                w->bef[c] += cv * (log(Cc[v] - cv + e) - lb); w->aft[c] += cv * (log(Cc[v] + w->ci[v] - cv + e) - la); }
        }
        for (int u = 0; u < ntz; u++) {
            const int c = w->tz[u]; double own = 0; const double lo = log(w->Nz[c] + S * e); const double *Cc = w->Cz + (int64_t)c * S;
            for (int q = w->cptr[i]; q < w->cptr[i + 1]; q++) own += w->cnt[q] * (log(Cc[w->sym[q]] + e) - lo);
            if (!w->ind[c]) { w->ind[c] = 1; w->td[w->ntd++] = c; w->dl[c] = 0; }
            w->dl[c] += w->aft[c] + own - w->bef[c] + w->n[i] * lS; w->seen[c] = 0;
        }
        for (int b = w->zptr[z]; b < w->zptr[z + 1]; b++) {
            const int j = w->zcell[b]; if (j == i) continue; const int c = cls[w->xc[j]]; if (c < 0) continue;
            for (int q = w->cptr[j]; q < w->cptr[j + 1]; q++) w->Cz[(int64_t)c * S + w->sym[q]] = 0;
        }
        for (int q = w->cptr[i]; q < w->cptr[i + 1]; q++) w->ci[w->sym[q]] = 0;
    }
}
static int best(W *w) {                                    /* the class that raises J most (ties to the lowest label); -1 if none raises it */
    int bc = -1; double bd = 1e-9;
    for (int u = 0; u < w->ntd; u++) { const int c = w->td[u]; if (w->dl[c] > bd + 1e-9 || (w->dl[c] > bd - 1e-9 && bc >= 0 && c < bc)) { if (w->dl[c] > 1e-9) { bd = w->dl[c]; bc = c; } } w->ind[c] = 0; }
    return bc;
}
double j_total(int m, int nx, int nz, int S, const int32_t *xc, const int32_t *zptr, const int32_t *zcell, const int32_t *cptr, const int32_t *sym,
               const float *cnt, const float *n, double e, const int32_t *cls) {
    double *Cz = calloc((size_t)nx * S, sizeof(double)), *Nz = calloc(nx, sizeof(double)), J = 0;
    for (int z = 0; z < nz; z++) {
        for (int b = zptr[z]; b < zptr[z + 1]; b++) { const int j = zcell[b], c = cls[xc[j]]; Nz[c] += n[j]; for (int q = cptr[j]; q < cptr[j + 1]; q++) Cz[(int64_t)c * S + sym[q]] += cnt[q]; }
        for (int b = zptr[z]; b < zptr[z + 1]; b++) { const int j = zcell[b], c = cls[xc[j]]; const double lb = log(Nz[c] - n[j] + S * e);
            for (int q = cptr[j]; q < cptr[j + 1]; q++) J += cnt[q] * (log(Cz[(int64_t)c * S + sym[q]] - cnt[q] + e) - lb); }
        for (int b = zptr[z]; b < zptr[z + 1]; b++) { const int j = zcell[b], c = cls[xc[j]]; Nz[c] = 0; for (int q = cptr[j]; q < cptr[j + 1]; q++) Cz[(int64_t)c * S + sym[q]] = 0; }
    }
    free(Cz); free(Nz); return J;
}
void find_classes(int m, int nx, int nz, int S, const int32_t *xc, const int32_t *zc, const int32_t *zptr, const int32_t *zcell, const int32_t *xptr, const int32_t *xcell,
                  const int32_t *cptr, const int32_t *sym, const float *cnt, const float *n, double e, const int32_t *xord, int sweeps, int dissolve, int32_t *cls) {
    W w = { m, nx, S, xc, zc, zptr, zcell, xptr, xcell, cptr, sym, cnt, n, e }; (void)nz;
    w.Cz = calloc((size_t)nx * S, sizeof(double)); w.Nz = calloc(nx, sizeof(double)); w.bef = calloc(nx, sizeof(double)); w.aft = calloc(nx, sizeof(double));
    w.dl = calloc(nx, sizeof(double)); w.ci = calloc(S, sizeof(double)); w.tz = calloc(nx, sizeof(int32_t)); w.td = calloc(nx, sizeof(int32_t)); w.seen = calloc(nx, 1); w.ind = calloc(nx, 1);
    int32_t *size = calloc(nx, sizeof(int32_t));                         /* a class that is founded takes the lowest label that has no member */
#define LOWEST(c) { c = 0; while (size[c]) c++; }
#define SWEEP for (int o = 0; o < nx; o++) { const int x = xord[o], old = cls[x]; size[old]--; cls[x] = -1; delta_all(&w, x, cls); int c = best(&w); \
        if (c < 0) { if (size[old] == 0) c = old; else LOWEST(c) } cls[x] = c; size[c]++; }
    for (int x = 0; x < nx; x++) cls[x] = -1;
    for (int o = 0; o < nx; o++) { const int x = xord[o]; delta_all(&w, x, cls); int c = best(&w); if (c < 0) LOWEST(c) cls[x] = c; size[c]++; }
    for (int sw = 0; sw < sweeps; sw++) SWEEP
    if (dissolve == 2) {  /* PLAN_8.md: a class is split around its worst-foretold member: that member founds a class, every other member chooses between the two; kept only if J rises; up to four times a class; then one more sweep */
        int32_t *save = malloc(nx * sizeof(int32_t)), *ssize = malloc(nx * sizeof(int32_t)), *mem = malloc(nx * sizeof(int32_t)), *labs = malloc(nx * sizeof(int32_t)); int nl = 0;
        double *xn = calloc(nx, sizeof(double)); for (int i = 0; i < m; i++) xn[xc[i]] += n[i];
        for (int c = 0; c < nx; c++) if (size[c] >= 3) labs[nl++] = c;
        for (int u = 0; u < nl; u++) { const int c0 = labs[u];
            for (int round = 0; round < 4 && size[c0] >= 3; round++) {
                const double Jb = j_total(m, nx, nz, S, xc, zptr, zcell, cptr, sym, cnt, n, e, cls); int nm = 0, wm = -1; double worst = 1e300;
                memcpy(save, cls, nx * sizeof(int32_t)); memcpy(ssize, size, nx * sizeof(int32_t));
                for (int o = 0; o < nx; o++) if (cls[xord[o]] == c0) mem[nm++] = xord[o];
                for (int i = 0; i < nm; i++) { const int x = mem[i]; cls[x] = -1; delta_all(&w, x, cls); double d = 0; for (int q = 0; q < w.ntd; q++) { if (w.td[q] == c0) d = w.dl[c0]; w.ind[w.td[q]] = 0; }
                    cls[x] = c0; d /= xn[x]; if (d < worst - 1e-9) { worst = d; wm = x; } }
                int L; size[c0]--; LOWEST(L) cls[wm] = L; size[L] = 1;
                for (int i = 0; i < nm; i++) { const int x = mem[i]; if (x == wm) continue; size[c0]--; cls[x] = -1; delta_all(&w, x, cls); double d0 = 0, dL = 0;
                    for (int q = 0; q < w.ntd; q++) { if (w.td[q] == c0) d0 = w.dl[c0]; if (w.td[q] == L) dL = w.dl[L]; w.ind[w.td[q]] = 0; }
                    const int c = dL > d0 + 1e-9 ? L : c0; cls[x] = c; size[c]++; }
                if (j_total(m, nx, nz, S, xc, zptr, zcell, cptr, sym, cnt, n, e, cls) < Jb + 1e-9) { memcpy(cls, save, nx * sizeof(int32_t)); memcpy(size, ssize, nx * sizeof(int32_t)); break; }
            }
        }
        SWEEP
        free(save); free(ssize); free(mem); free(labs); free(xn);
    }
    if (dissolve == 1) {  /* each class of two or more is dissolved once and its members seated again one by one; kept only if J rises; then one more sweep */
        int32_t *save = malloc(nx * sizeof(int32_t)), *ssize = malloc(nx * sizeof(int32_t)), *mem = malloc(nx * sizeof(int32_t)), *labs = malloc(nx * sizeof(int32_t)); int nl = 0;
        for (int c = 0; c < nx; c++) if (size[c] >= 2) labs[nl++] = c;
        for (int u = 0; u < nl; u++) { const int c0 = labs[u]; if (size[c0] < 2) continue;
            const double Jb = j_total(m, nx, nz, S, xc, zptr, zcell, cptr, sym, cnt, n, e, cls); int nm = 0;
            memcpy(save, cls, nx * sizeof(int32_t)); memcpy(ssize, size, nx * sizeof(int32_t));
            for (int o = 0; o < nx; o++) if (cls[xord[o]] == c0) { mem[nm++] = xord[o]; cls[xord[o]] = -1; }
            size[c0] = 0;
            for (int i = 0; i < nm; i++) { const int x = mem[i]; delta_all(&w, x, cls); int c = best(&w); if (c < 0) LOWEST(c) cls[x] = c; size[c]++; }
            if (j_total(m, nx, nz, S, xc, zptr, zcell, cptr, sym, cnt, n, e, cls) < Jb + 1e-9) { memcpy(cls, save, nx * sizeof(int32_t)); memcpy(size, ssize, nx * sizeof(int32_t)); }
        }
        SWEEP
        free(save); free(ssize); free(mem); free(labs);
    }
    free(w.Cz); free(w.Nz); free(w.bef); free(w.aft); free(w.dl); free(w.ci); free(w.tz); free(w.td); free(w.seen); free(w.ind); free(size);
}

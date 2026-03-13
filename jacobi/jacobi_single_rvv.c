#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include <riscv_vector.h>

#define DATATYPE float

// Boundary value at the LHS of the bar
#define LEFT_VALUE  1.0f
// Boundary value at the RHS of the bar
#define RIGHT_VALUE 10.0f
// The maximum number of iterations
#define MAX_ITERATIONS 100000
// How often to report the norm
#define REPORT_NORM_PERIOD 100

static void initialise(DATATYPE*, DATATYPE*, int, int);
static uint64_t millis();

/*
 * Compute sum of squared residuals over the interior domain.
 *
 * residual(i,j) = 4*u[i,j] - u[i-1,j] - u[i+1,j] - u[i,j-1] - u[i,j+1]
 *
 * The inner loop (i = 1..ny) is contiguous in memory, so we use
 * unit-stride vle32 loads.  LMUL=4 gives 4x the natural vector length;
 * with VLEN=256 and f32 this is 32 floats per vector group.
 */
static float compute_rnorm(const DATATYPE *u_k, int nx, int ny, int msz_y) {
    float rnorm = 0.0f;

    for (int j = 1; j <= nx; j++) {
        const float *center = &u_k[1 +  j      * msz_y];
        const float *left   = &u_k[0 +  j      * msz_y]; /* i-1 */
        const float *right  = &u_k[2 +  j      * msz_y]; /* i+1 */
        const float *up     = &u_k[1 + (j - 1) * msz_y]; /* j-1 */
        const float *down   = &u_k[1 + (j + 1) * msz_y]; /* j+1 */

        size_t n = (size_t)ny;
        size_t i = 0;
        while (i < n) {
            size_t vl = __riscv_vsetvl_e32m4(n - i);

            vfloat32m4_t vcenter = __riscv_vle32_v_f32m4(center + i, vl);
            vfloat32m4_t vleft   = __riscv_vle32_v_f32m4(left   + i, vl);
            vfloat32m4_t vright  = __riscv_vle32_v_f32m4(right  + i, vl);
            vfloat32m4_t vup     = __riscv_vle32_v_f32m4(up     + i, vl);
            vfloat32m4_t vdown   = __riscv_vle32_v_f32m4(down   + i, vl);

            /* neighbors = left + right + up + down */
            vfloat32m4_t neighbors = __riscv_vfadd_vv_f32m4(vleft,     vright, vl);
            neighbors              = __riscv_vfadd_vv_f32m4(neighbors,  vup,   vl);
            neighbors              = __riscv_vfadd_vv_f32m4(neighbors,  vdown, vl);

            /* residual = 4*center - neighbors
             * vfmacc: acc = acc + scalar * v
             *   start with acc = -neighbors, then add 4.0 * center */
            vfloat32m4_t neg_nb   = __riscv_vfneg_v_f32m4(neighbors, vl);
            vfloat32m4_t residual = __riscv_vfmacc_vf_f32m4(neg_nb, 4.0f, vcenter, vl);

            /* sq = residual^2 */
            vfloat32m4_t sq = __riscv_vfmul_vv_f32m4(residual, residual, vl);

            /* horizontal reduction into scalar */
            vfloat32m1_t zero = __riscv_vfmv_s_f_f32m1(0.0f, 1);
            vfloat32m1_t red  = __riscv_vfredusum_vs_f32m4_f32m1(sq, zero, vl);
            float partial;
            __riscv_vse32_v_f32m1(&partial, red, 1);
            rnorm += partial;

            i += vl;
        }
    }
    return rnorm;
}

/*
 * Jacobi update: u_kp1[i,j] = 0.25 * (u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1])
 *
 * Only the four neighbours are loaded — center is not read.
 * LMUL=8: with VLEN=256 and f32 this is 64 floats per vector group.
 * Register pressure is low (4 load vectors peak = 4×8 = 32 physical registers),
 * so m8 fills the vector register file without spilling.
 */
static void jacobi_update(const DATATYPE *u_k, DATATYPE *u_kp1,
                          int nx, int ny, int msz_y) {
    for (int j = 1; j <= nx; j++) {
        const float *left  = &u_k[0 +  j      * msz_y];
        const float *right = &u_k[2 +  j      * msz_y];
        const float *up    = &u_k[1 + (j - 1) * msz_y];
        const float *down  = &u_k[1 + (j + 1) * msz_y];
        float       *out   = &u_kp1[1 + j * msz_y];

        size_t n = (size_t)ny;
        size_t i = 0;
        while (i < n) {
            size_t vl = __riscv_vsetvl_e32m8(n - i);

            vfloat32m8_t vleft  = __riscv_vle32_v_f32m8(left  + i, vl);
            vfloat32m8_t vright = __riscv_vle32_v_f32m8(right + i, vl);
            vfloat32m8_t vup    = __riscv_vle32_v_f32m8(up    + i, vl);
            vfloat32m8_t vdown  = __riscv_vle32_v_f32m8(down  + i, vl);

            /* sum = left + right + up + down */
            vfloat32m8_t sum = __riscv_vfadd_vv_f32m8(vleft, vright, vl);
            sum              = __riscv_vfadd_vv_f32m8(sum,   vup,    vl);
            sum              = __riscv_vfadd_vv_f32m8(sum,   vdown,  vl);

            /* result = 0.25 * sum */
            vfloat32m8_t result = __riscv_vfmul_vf_f32m8(sum, 0.25f, vl);

            __riscv_vse32_v_f32m8(out + i, result, vl);
            i += vl;
        }
    }
}

int main(int argc, char *argv[]) {
    int nx, ny, max_its;
    float convergence_accuracy;

    if (argc != 5) {
        printf("You should provide four command line arguments, the global size in X, "
               "the global size in Y, convergence accuracy and max number iterations\n");
        printf("In the absence of this defaulting to x=128, y=1024, convergence=3e-3, "
               "no max number of iterations\n");
        nx = 128;
        ny = 1024;
        convergence_accuracy = 3e-3f;
        max_its = 0;
    } else {
        nx = atoi(argv[1]);
        ny = atoi(argv[2]);
        convergence_accuracy = (float)atof(argv[3]);
        max_its = atoi(argv[4]);
    }

#ifdef INSTRUMENTED
    if (max_its < 1 || max_its > 100) {
        max_its = 100;
        printf("Limiting the instrumented run to 100 iterations to keep file size small, "
               "you can change this in the code if you really want\n");
    }
#endif

    printf("Global size in X=%d, Global size in Y=%d\n\n", nx, ny);

    int msz_y = ny + 2;
    int msz_x = nx + 2;

    DATATYPE *u_k   = malloc(sizeof(DATATYPE) * msz_x * msz_y);
    DATATYPE *u_kp1 = malloc(sizeof(DATATYPE) * msz_x * msz_y);
    DATATYPE *temp;

    initialise(u_k, u_kp1, nx, ny);

    float bnorm = sqrtf(compute_rnorm(u_k, nx, ny, msz_y));
    float norm  = 0.0f;

    int k;
    uint64_t start_ms = millis();
    for (k = 0; k < MAX_ITERATIONS; k++) {
        float rnorm = compute_rnorm(u_k, nx, ny, msz_y);
        norm = sqrtf(rnorm) / bnorm;

        if (norm < convergence_accuracy) break;
        if (max_its > 0 && k >= max_its) break;

        jacobi_update(u_k, u_kp1, nx, ny, msz_y);

        temp = u_kp1; u_kp1 = u_k; u_k = temp;

        if (k % REPORT_NORM_PERIOD == 0)
            printf("Iteration= %d Relative Norm=%e\n", k, norm);
    }
    uint64_t end_ms = millis();
    printf("\nTerminated on %d iterations, Relative Norm=%e, Total time=%llu ms\n",
           k, norm, (unsigned long long)(end_ms - start_ms));

    free(u_k);
    free(u_kp1);
    return 0;
}

/**
 * Initialises the arrays, such that u_k contains the boundary conditions at the
 * start and end points and all other points are zero.  u_kp1 is set to equal u_k.
 */
static void initialise(DATATYPE *u_k, DATATYPE *u_kp1, int nx, int ny) {
    int i, j;
    for (i = 0; i < nx + 1; i++) {
        u_k[i * (ny + 2)]            = LEFT_VALUE;
        u_k[(ny + 1) + i * (ny + 2)] = RIGHT_VALUE;
    }
    for (j = 0; j <= nx + 1; j++) {
        for (i = 1; i <= ny; i++) {
            u_k[i + j * (ny + 2)] = 0.0f;
        }
    }
    for (j = 0; j <= nx + 1; j++) {
        for (i = 0; i <= ny + 1; i++) {
            u_kp1[i + j * (ny + 2)] = u_k[i + j * (ny + 2)];
        }
    }
}

static uint64_t millis() {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        perror("clock_gettime");
        return 0;
    }
    return (uint64_t)ts.tv_sec * 1000ULL + ts.tv_nsec / 1000000ULL;
}

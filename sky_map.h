#include <gsl/gsl_math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sort_double.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_fft_real.h>
#include <gsl/gsl_fft_halfcomplex.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_heapsort.h>
#include <gsl/gsl_sort_vector.h>
#include "IMRPhenomD.h"

#define NX 7           // number of quasi-intrinsic parameters (Mc, Mt, chi1, chi2, phic, tc, DL)
#define NP 11          // total number of entries in the full parameter arrays NX+4
#define NS 7           // number of quasi-extrinsic parameters (alpha, delta, psi, cos(iota), scale, phi0=2*phic, dt)
#define NC 16           // number of chains
#define fref 100.0       // reference freqency for PhenomD waveform
#define h22fac  0.31539156525252               //  2. * sqrt(5. / (64.*PI)) factor for h22 to h conversion
#define TSUN 4.92569043916e-6                                           // mass to seconds conversion
#define CLIGHT 299792458.0                     // m/s
#define dtmax 0.03            // Maximum time shift relative to geocenter waveform in sky likelihood
#define TPI 6.2831853071795862319959269370884
#define DLmax 1.0e10
#define DLmin 1.0e6
#ifndef DETECTOR_CATALOG_SIZE
#define DETECTOR_CATALOG_SIZE 4
#endif
#ifndef DEFAULT_NIFO
#define DEFAULT_NIFO 4
#endif
#define SKYMAP_NSIDE 256
#ifndef BNS_SNR_THRESHOLD
#define BNS_SNR_THRESHOLD 8.0
#endif
#ifndef BNS_MAX_DETECTION_TRIES
#define BNS_MAX_DETECTION_TRIES 1000
#endif
// Local BNS merger-rate interval from GWTC-5.0, used to convert the
// Monte Carlo detection efficiency into detections per year.
#ifndef BNS_RATE_INTRINSIC_DRAWS
#define BNS_RATE_INTRINSIC_DRAWS 100
#endif
#ifndef BNS_RATE_ORIENTATION_DRAWS
#define BNS_RATE_ORIENTATION_DRAWS 1000
#endif
#ifndef BNS_RATE_DISTANCE_SPLINE_SIZE
#define BNS_RATE_DISTANCE_SPLINE_SIZE 512
#endif
#ifndef BNS_MERGER_RATE_LOW_GPC3_YR
#define BNS_MERGER_RATE_LOW_GPC3_YR 5.1
#endif
#ifndef BNS_MERGER_RATE_HIGH_GPC3_YR
#define BNS_MERGER_RATE_HIGH_GPC3_YR 154.7
#endif
// Settings for the LIGO-style BNS range diagnostic. The range is the
// volume-equivalent detection distance for a non-spinning 1.4+1.4 Msun binary,
// averaged over sky position, polarization, and inclination at the SNR threshold.
#ifndef BNS_RANGE_MC_DRAWS
#define BNS_RANGE_MC_DRAWS 100000
#endif
#ifndef BNS_RANGE_COMPONENT_MASS_MSUN
#define BNS_RANGE_COMPONENT_MASS_MSUN 1.4
#endif
#ifndef BNS_RANGE_REFERENCE_DISTANCE_MPC
#define BNS_RANGE_REFERENCE_DISTANCE_MPC 100.0
#endif
// Catalog sources can be more massive than the canonical 1.4+1.4 Msun BNS
// used for the range diagnostic. Their amplitude scales roughly as Mc^(5/6),
// so pad the population distance cutoff to avoid clipping louder high-mass draws.
#ifndef BNS_POPULATION_DISTANCE_SCALE
#define BNS_POPULATION_DISTANCE_SCALE 1.4
#endif
#ifndef BNS_MASS_MEAN_MSUN
#define BNS_MASS_MEAN_MSUN 1.5
#endif
#ifndef BNS_MASS_SIGMA_MSUN
#define BNS_MASS_SIGMA_MSUN 0.2
#endif
#ifndef BNS_MASS_MIN_MSUN
#define BNS_MASS_MIN_MSUN 1.1
#endif
#ifndef BNS_MASS_MAX_MSUN
#define BNS_MASS_MAX_MSUN 2.5
#endif
#ifndef BNS_SPIN_MAX
#define BNS_SPIN_MAX 0.1
#endif
#define LCDM_H0_KM_S_MPC 67.4
#define LCDM_OMEGA_M 0.315
#define LCDM_OMEGA_L 0.685
#ifndef LOG_LIKELIHOOD_FULL_VERBOSE
#define LOG_LIKELIHOOD_FULL_VERBOSE 0
#endif
#ifndef PRINT_MCMC_STATE
#define PRINT_MCMC_STATE 0
#endif
#ifndef POPULATION_HISTOGRAM_BINS
#define POPULATION_HISTOGRAM_BINS 50
#endif
#define FAST_SNR_GRID_SIZE 1024
#define USE_FAST_DETECTION_SNR 1
#ifndef FAST_SNR_EXACT_CHECK
#define FAST_SNR_EXACT_CHECK 0
#endif

struct Net
{
    int Nifo;
    double Tobs;
    double offset;
    double ttrig;
    double tmax;
    double tmin;
    double GMST;
    int *labels;
    double *tds;
    double **delays;
    int *lstart;
    int *lstop;
};

typedef struct {
    double x, y, z;
} Vector3D;

// Structure to encapsulate unique parameters for each detector site
typedef struct {
    const char* name;
    Vector3D vertex; // ECEF Vertex coordinates in meters
    Vector3D arm_X;  // ECEF Normalized X-arm vector
    Vector3D arm_Y;  // ECEF Normalized Y-arm vector
} Detector;


double fbegin(double *param);
void geotemplate(double *gwave, RealVector *freq, double *params, int N);
Vector3D normalize(Vector3D v);
double dot_product(Vector3D a, Vector3D b);
double compute_response(Vector3D X, Vector3D Y, Vector3D P, Vector3D Q);
double compute_cross_response(Vector3D X, Vector3D Y, Vector3D P, Vector3D Q);
void setup_network(Detector *network);
void Response(Detector *network, double ra_rad, double dec_rad, double psi_rad, double gmst, double *Fp, double *Fc, double *delay);
double compute_geocenter_delay(Vector3D vertex, Vector3D wave_prop);
void load_psd(const char *filename, double *Sn, double *freqs, int nfreq, double fmin, double fmax);
void fulltemplates(struct Net *net, Detector *network, double **hwave, RealVector *freq, double *params, int N);
void skylikesetup(struct Net *net, double **data,  double **wave, double *D, double *H, double **DHc,  double **DHs, double Tobs, int n, int nt);
void fisherskysetup(struct Net *net, double **wave, double **HH, double Tobs, int n);
double skylike(struct Net *net, Detector *network, double *params, double *D, double *H, double **DHc,  double **DHs, double dt, int nt, int flag);
double f_nwip(double *a, double *b, int n);
double log_likelihood_full(struct Net *net, Detector *network, double **D, double **SN, double *params, RealVector *freq, int N, double Tobs);
double fourier_nwip(double *a, double *b, double *Sn, int imin, int imax, int N);
void skymcmc(struct Net *net, Detector *network, int MCX, int *mxc, FILE *chain, double **paramx, double **skyx, double **pallx, int *who, double *heat, double dtx, int nt, double *DD, double **WW, double ***DHc,  double ***DHs, double ***HH, double Tobs, gsl_rng * r);
void fisher_matrix_fastsky(struct Net *net, Detector *network, double *params, double **fisher, double **HH);
void fisher_skyproposal(gsl_rng * r, double **skyvecs, double *skyevals, double *jump);
void FisherEvec(double **fish, double *eval, double **evec, int d);
void Ring(Detector *network, double *skyx, double *skyy, int d1, int d2, double GMST, gsl_rng * r);
void DetectorPlaneAntipode(Detector *network, double *skyx, double *skyy, int d1, int d2, int d3, double GMST);
int skymap(Detector *network, double *paramsx, double *paramsy, double GMST, int ifo1, int ifo2, int iref);
double skydensity(Detector *network, double *paramsx, double *paramsy, double GMST, int ifo1, int ifo2, int iref);
void uvwz(double *u, double *v, double *w, double *z, double *params);
void uvwz_sol(double *uy, double *vy, double *wy, double *zy, double ux, double vx, double wx, double zx, double fp1x, double fp1y, double fc1x, double fc1y, double fp2x, double fp2y, double fc2x, double fc2y);
void exsolve(double *phiy, double *psiy, double *Ay, double *ciotay, double uy, double vy, double wy, double zy);
void dshifts(struct Net *net, Detector *network, double *sky, double *params);
double det(double *A, int N);


int *int_vector(int N);
void free_int_vector(int *v);
double *double_vector(int N);
void free_double_vector(double *v);
double **double_matrix(int N, int M);
void free_double_matrix(double **m, int N);
double ***double_tensor(int N, int M, int L);
void free_double_tensor(double ***t, int N, int M);
int **int_matrix(int N, int M);
void free_int_matrix(int **m, int N);
double ****double_quad(int N, int M, int L, int K);
void free_double_quad(double ****t, int N, int M, int L);

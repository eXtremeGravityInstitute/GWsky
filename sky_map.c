/*******************************************************************************************

Copyright (c) 2026 Neil Cornish

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

**********************************************************************************************/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <chealpix.h>
#include "sky_map.h"

//##############################################
//MT modifications

gsl_rng **rvec;
//##############################################

typedef struct {
    double area50;
    double area90;
    double boundary_area;
    long boundary_pixels;
    double map_theta;
    double map_phi;
} SkyPostSummary;

static void initialize_chain_state(struct Net *net, double *params, double **paramx, double **pallx, double **skyx, int *who, double *heat, int *mxc);
static void prepare_internal_params(double *params, const double *file_params, double Tobs, double offset);
static int read_injection_file(const char *filename, double *file_params);
static void write_injection_file(const char *filename, const double *file_params);
static void draw_bns_population(double *file_params, double Tobs, double offset, double max_luminosity_distance_mpc, gsl_rng *r);
static double draw_bns_mass(gsl_rng *r);
static double lcdm_eofz(double z);
static double lcdm_comoving_distance_mpc(double z);
static double lcdm_redshift_from_comoving_distance(double dc_mpc);
static double lcdm_redshift_from_luminosity_distance(double dl_mpc);
static double lcdm_luminosity_distance_from_comoving_distance(double dc_mpc);
static double compute_detection_snr(struct Net *net, Detector *network, double **D, double **SN, double *params, RealVector *freq, int N, double Tobs, double *detector_snr);
static double compute_exact_detection_snr(struct Net *net, Detector *network, double **D, double **SN, double *params, RealVector *freq, int N, double Tobs, double *detector_snr);
static double fast_network_snr2(struct Net *net, Detector *network, double **SN, double *params, double Tobs, int N, double *detector_snr);
static int fast_detector_snr2_norms(struct Net *net, double **SN, double *params, double Tobs, int N, double *detector_norm2);
static double estimate_bns_range(struct Net *net, Detector *network, double **SN, int N, double Tobs, gsl_rng *r, double *mean_threshold_distance, double *max_threshold_distance);
static double interpolate_uniform_spectrum(double *y, int n, double Tobs, double f);
static void setup_source_likelihood(struct Net *net, Detector *network, double *params, double **D, double **SN, double **data, double *hwave, double ***wave, double *DD, double **WW, double ***DHc, double ***DHs, double ***HH, double **skyx, double **paramx, double **pallx, int *who, double *heat, int *mxc, double *logLsky, RealVector *freq, int N, int nt, double Tobs);
static int run_source_mcmc(struct Net *net, Detector *network, double *params, double **D, double **SN, double **data, double *hwave, double ***wave, double *DD, double **WW, double ***DHc, double ***DHs, double ***HH, double **skyx, double **paramx, double **pallx, int *who, double *heat, int *mxc, double *logLsky, RealVector *freq, int N, int nt, double Tobs, double dt, int Nsky, gsl_rng *r, SkyPostSummary *summary);
static int ensure_directory(const char *path);
static int move_file_to_source_name(const char *source, const char *dest_dir, const char *prefix, int source_id);
static int write_summary_file(const char *filename, int source_id, int attempts, const SkyPostSummary *summary, struct Net *net, Detector *network, const double *detector_snr, double network_snr);
static int write_population_histograms(const char *dest_dir, const double *area50, const double *area90, const double *distance, const double *snr, int n, double distance_max_mpc);
static int write_histogram_file(const char *filename, const char *quantity, const char *units, const double *values, int n, int nbins, double xmin, double xmax);
static double finite_array_max(const double *values, int n);
static int postprocess_sky_chain(const char *chain_file, long Nside, SkyPostSummary *summary);
static int parameter_array_length(struct Net *net);
static int skyhist_is_power_of_two(long n);
static double skyhist_clamp_double(double x, double xmin, double xmax);
static int skyhist_write_map(const char *filename, double *map, long Npix);
static long skyhist_make_credible_boundary(long Nside, long Npix, double *map, gsl_permutation *perm, long k90, double level90, double *boundary, double *region);
static void skyhist_nested_neighbors(long nside, long ipnest, long neighbors[8]);
static void skyhist_nest2xyf(long nside, long ipnest, long *ix, long *iy, long *face);
static long skyhist_xyf2nest(long nside, long ix, long iy, long face);
static long skyhist_edge_neighbor_nest(long nside, long ix, long iy, long face, int dir);
static long skyhist_mod4(long x);
static double *skyhist_double_vector(long N);
static int *skyhist_int_vector(long N);

// gcc -I/opt/homebrew/include -L/opt/homebrew/lib -o sky_map sky_map.c IMRPhenomD_internals.c IMRPhenomD.c -lm -lgslcblas -lgsl -lchealpix

int main(int argc, char *argv[])
{
    int i, j, N;
    int id, ifo, nt, param_count;
    double fmin, fmax, Tobs;
    double x, y, dt;
    double **SN, *freqs;
    double ***wave;
    double *params;
    double *hwave;
    double **data, **D;
    double *DD;
    double **WW;
    double ***DHc, ***DHs, ***HH;
    double **skyx;
    double **paramx, **pallx;
    double *logLsky;
    int *who, *mxc;
    double *heat;
    char command[1024];
    const char **psd_file;
    Detector network[DETECTOR_CATALOG_SIZE];
    
    struct Net *net  = malloc(sizeof(struct Net));
    
    FILE *in;
    FILE *chain;
    const gsl_rng_type *P;
    gsl_rng *r, *range_rng;
    
    //##############################################
    //open MP modifications
    //omp_set_num_threads(8);
    
    gsl_rng_env_setup();
    P = gsl_rng_default;
    r = gsl_rng_alloc(P);
    gsl_rng_set(r, 0);
    
    rvec = (gsl_rng **)malloc(sizeof(gsl_rng *) * (NC+1));
    for(i = 0 ; i<= NC; i++){
        rvec[i] = gsl_rng_alloc(P);
        gsl_rng_set(rvec[i] , i);
    }
    //##############################################
    
    who = int_vector(NC+1);
    heat = double_vector(NC+1);
    mxc = int_vector(3);
    
     // intialize the who, heat, history and counter arrays
     for(i=1; i<=NC; i++) who[i] = i;
     heat[1] = 1.0;
     for(i=2; i<=NC; i++) heat[i] = heat[i-1]*1.15;  // hot chains
    
    for (i = 0; i < 3; i++) mxc[i] = 0;
    
    fmin = 10.0;
    fmax = 4096.0;
    Tobs = 1024.0;
    
    // Hour angle. Set to zero so Detector frame is Earth frame
    // If you want to run on real events, you will need to use the GPS to GMST mappings
    net->GMST = 0.0;
    net->Tobs = Tobs;
    net->offset = 4.0;
    
    // Active detector count. The labels array maps active slots onto
    // the detector catalog initialized by setup_network().
    net->Nifo = DEFAULT_NIFO;
    
    if(net->Nifo < 1 || net->Nifo > DETECTOR_CATALOG_SIZE)
    {
        fprintf(stderr, "Invalid number of detectors: Nifo=%d, catalog size=%d\n", net->Nifo, DETECTOR_CATALOG_SIZE);
        exit(1);
    }
    
    net->labels = int_vector(net->Nifo);
    
    // This is the standard order, H1, L1, A1 and Virgo. To run e.g. just L1 and Virgo
    // set net->Nifo = 2 and net_labels[0] = 1 and net_labels[1] = 3 etc. You also need
    // to change DEFAULT_NIFO to 2 in this example.
    for (i = 0; i < net->Nifo; ++i) net->labels[i] = i;
    
    setup_network(network);
    for (i = 0; i < net->Nifo; ++i)
    {
        if(net->labels[i] < 0 || net->labels[i] >= DETECTOR_CATALOG_SIZE)
        {
            fprintf(stderr, "Detector label %d is outside the catalog range [0,%d)\n", net->labels[i], DETECTOR_CATALOG_SIZE);
            exit(1);
        }
    }
    
    net->tds = double_vector(net->Nifo);
    net->delays = double_matrix(net->Nifo,net->Nifo);
    
    
    // number of data samples
    N = (int)(2.0*fmax*Tobs);
    
    dt = Tobs/(double)(N);
    
    hwave = double_vector(N);
    // whitened template in each detector
    wave = double_tensor(NC+1,net->Nifo,N);
    param_count = parameter_array_length(net);
    paramx = double_matrix(NC+1, param_count);
    pallx = double_matrix(NC+1, param_count);
    SN = double_matrix(net->Nifo,N/2);  // PSDs
    freqs = double_vector(N/2);  // frequencies;
    RealVector *freq;
    freq = CreateRealVector((N/2));
    psd_file = (const char **)malloc(sizeof(char *)*net->Nifo);
    D = double_matrix(net->Nifo,N);
    data = double_matrix(net->Nifo,N); // whitened data
    
    // allow for geocenter - surface light travel time
    // and for attempts to slide the waveform by the max delay time
    nt = 2*(int)(dtmax/dt)+1;
    // these hold the various inner products used in the fast sky likelihood
    DD = double_vector(net->Nifo);
    WW = double_matrix(NC+1,net->Nifo);
    DHc = double_tensor(NC+1,net->Nifo,nt);
    DHs = double_tensor(NC+1,net->Nifo,nt);
    HH = double_tensor(NC+1,net->Nifo,3);
    skyx = double_matrix(NC+1,7);
    logLsky = double_vector(NC+1);
    params = double_vector(NP);
    for (i = 0; i < N/2; ++i) freqs[i] = (double)(i)/Tobs;
    
 
    freq->data[0] = 1.0/Tobs;
    for (i=1; i< N/2; i++) freq->data[i] = freqs[i];
    
    for (id = 0; id < net->Nifo; ++id)
    {
        ifo = net->labels[id];
        if(ifo == 2) psd_file[id] = "aligo_O4high.txt";  // initial LIGO India performance; ASD file from LIGO DCC T2200043: https://dcc.ligo.org/LIGO-T2200043/public
        else if(ifo == 3) psd_file[id] = "avirgo_O5high_NEW.txt";  // Virgo O5 high-noise conservative estimate, LIGO DCC T2200043-v3: https://dcc.ligo.org/public/0180/T2200043/003/
        else psd_file[id] = "AplusDesign.txt";           // US detector reference sensitivity; ASD file from LIGO DCC T2200043: https://dcc.ligo.org/LIGO-T2200043/public
    }

    for (id = 0; id < net->Nifo; ++id)
    {
        load_psd(psd_file[id], SN[id], freqs, N/2, fmin, fmax);
    }
    
    int catalog_sources = 0;
    int catalog_mode = 0;
    int Nsky = 200000;
    int source_id, attempts;
    double file_params[NP];
    double detector_snr[DETECTOR_CATALOG_SIZE];
    double network_snr, bns_range, bns_mean_distance, bns_max_distance, catalog_max_distance;
    double *catalog_area50, *catalog_area90, *catalog_distance, *catalog_snr;
    SkyPostSummary sky_summary;
    char filename[1024];
    catalog_area50 = NULL;
    catalog_area90 = NULL;
    catalog_distance = NULL;
    catalog_snr = NULL;
    
    if(argc > 1) catalog_sources = atoi(argv[1]);
    catalog_mode = (catalog_sources > 0);
    
    range_rng = gsl_rng_alloc(P);
    if(range_rng == NULL)
    {
        fprintf(stderr, "Could not allocate BNS range random number generator\n");
        return 1;
    }
    gsl_rng_set(range_rng, 7919);
    bns_range = estimate_bns_range(net, network, SN, N, Tobs, range_rng, &bns_mean_distance, &bns_max_distance);
    gsl_rng_free(range_rng);
    if(bns_range < 0.0) return 1;
    
    printf("BNS network range = %.2f Mpc using %d orientation draws, SNR threshold %.1f\n",
           bns_range, BNS_RANGE_MC_DRAWS, BNS_SNR_THRESHOLD);
    printf("BNS arithmetic mean threshold distance = %.2f Mpc; max sampled threshold distance = %.2f Mpc\n",
           bns_mean_distance, bns_max_distance);
    catalog_max_distance = BNS_POPULATION_DISTANCE_SCALE*bns_max_distance;
    
    if(!catalog_mode)
    {
        printf("Single-injection mode: reading injection.dat\n");
        if(read_injection_file("injection.dat", file_params) != 0) return 1;
        prepare_internal_params(params, file_params, Tobs, net->offset);
        network_snr = compute_detection_snr(net, network, D, SN, params, freq, N, Tobs, detector_snr);
        printf("Network SNR = %f\n", network_snr);
        for(id = 0; id < net->Nifo; ++id)
        {
            ifo = net->labels[id];
            printf("SNR in detector %s %e\n", network[ifo].name, detector_snr[id]);
        }
        
        if(run_source_mcmc(net, network, params, D, SN, data, hwave, wave, DD, WW, DHc, DHs, HH, skyx, paramx, pallx, who, heat, mxc, logLsky, freq, N, nt, Tobs, dt, Nsky, r, &sky_summary) != 0)
        {
            return 1;
        }
        
        // Render the maps produced by postprocess_sky_chain().
        sprintf(command, "sh sky.sh 0");
        system(command);
    }
    else
    {
        printf("Catalog mode: generating %d detectable BNS sources with network SNR >= %.2f\n", catalog_sources, BNS_SNR_THRESHOLD);
        printf("Catalog population maximum luminosity distance set to %.2f Mpc = %.2f x %.2f Mpc range-diagnostic maximum\n",
               catalog_max_distance, BNS_POPULATION_DISTANCE_SCALE, bns_max_distance);
        if(ensure_directory("skymaps") != 0) return 1;
        catalog_area50 = double_vector(catalog_sources);
        catalog_area90 = double_vector(catalog_sources);
        catalog_distance = double_vector(catalog_sources);
        catalog_snr = double_vector(catalog_sources);
        if(catalog_area50 == NULL || catalog_area90 == NULL || catalog_distance == NULL || catalog_snr == NULL)
        {
            fprintf(stderr, "Could not allocate population histogram arrays\n");
            return 1;
        }
        
        for(source_id = 1; source_id <= catalog_sources; ++source_id)
        {
            network_snr = 0.0;
            for(attempts = 1; attempts <= BNS_MAX_DETECTION_TRIES; ++attempts)
            {
                draw_bns_population(file_params, Tobs, net->offset, catalog_max_distance, r);
                prepare_internal_params(params, file_params, Tobs, net->offset);
                network_snr = compute_detection_snr(net, network, D, SN, params, freq, N, Tobs, detector_snr);
                if(network_snr >= BNS_SNR_THRESHOLD) break;
            }
            
            if(network_snr < BNS_SNR_THRESHOLD)
            {
                fprintf(stderr, "Failed to generate a detectable BNS source after %d tries.\n", BNS_MAX_DETECTION_TRIES);
                fprintf(stderr, "The catalog distance limit was set to %.2f Mpc using %.2f times the BNS range diagnostic maximum.\n",
                        catalog_max_distance, BNS_POPULATION_DISTANCE_SCALE);
                fprintf(stderr, "If this persists, consider increasing BNS_MAX_DETECTION_TRIES or using a smaller distance limit.\n");
                return 1;
            }
            
            printf("Source %d accepted after %d population draws with network SNR = %f\n", source_id, attempts, network_snr);
            for(id = 0; id < net->Nifo; ++id)
            {
                ifo = net->labels[id];
                printf("SNR in detector %s %e\n", network[ifo].name, detector_snr[id]);
            }
            
            snprintf(filename, sizeof(filename), "skymaps/injection_%d.dat", source_id);
            write_injection_file(filename, file_params);
            
            if(run_source_mcmc(net, network, params, D, SN, data, hwave, wave, DD, WW, DHc, DHs, HH, skyx, paramx, pallx, who, heat, mxc, logLsky, freq, N, nt, Tobs, dt, Nsky, r, &sky_summary) != 0)
            {
                return 1;
            }
            
            snprintf(filename, sizeof(filename), "skymaps/summary_%d.dat", source_id);
            if(write_summary_file(filename, source_id, attempts, &sky_summary, net, network, detector_snr, network_snr) != 0) return 1;
            catalog_area50[source_id-1] = sky_summary.area50;
            catalog_area90[source_id-1] = sky_summary.area90;
            catalog_distance[source_id-1] = file_params[6];
            catalog_snr[source_id-1] = network_snr;
            if(move_file_to_source_name("sky.dat", "skymaps", "sky", source_id) != 0) return 1;
            if(move_file_to_source_name("sky90_region.dat", "skymaps", "sky90_region", source_id) != 0) return 1;
            if(move_file_to_source_name("sky90_boundary.dat", "skymaps", "sky90_boundary", source_id) != 0) return 1;
        }
        
        if(write_population_histograms("skymaps", catalog_area50, catalog_area90, catalog_distance, catalog_snr, catalog_sources, catalog_max_distance) != 0) return 1;
        free_double_vector(catalog_area50);
        free_double_vector(catalog_area90);
        free_double_vector(catalog_distance);
        free_double_vector(catalog_snr);
    }
    
    return 0;
}

static int parameter_array_length(struct Net *net)
{
    int n = NX + 3*net->Nifo;
    return (n > NP) ? n : NP;
}

static void initialize_chain_state(struct Net *net, double *params, double **paramx, double **pallx, double **skyx, int *who, double *heat, int *mxc)
{
    int i, j, nparam;
    
    nparam = parameter_array_length(net);
    
    for(i = 0; i < 3; i++) mxc[i] = 0;
    
    for(i = 1; i <= NC; i++) who[i] = i;
    heat[1] = 1.0;
    for(i = 2; i <= NC; i++) heat[i] = heat[i-1]*1.15;
    
    for(i = 1; i <= NC; ++i)
    {
        for(j = 0; j < nparam; ++j)
        {
            paramx[i][j] = 0.0;
            pallx[i][j] = 0.0;
        }
        for(j = 0; j < NP; ++j)
        {
            paramx[i][j] = params[j];
            pallx[i][j] = params[j];
        }
    }
    
    skyx[1][0] = params[7];
    skyx[1][1] = params[8];
    skyx[1][2] = params[9];
    skyx[1][3] = params[10];
    skyx[1][4] = 1.0;
    skyx[1][5] = 0.0;
    skyx[1][6] = 0.0;
    
    for(i = 2; i <= NC; i++)
    {
        for(j = 0; j < NS; ++j) skyx[i][j] = skyx[1][j];
    }
}

static void prepare_internal_params(double *params, const double *file_params, double Tobs, double offset)
{
    int i;
    
    for(i = 0; i < NP; ++i) params[i] = file_params[i];
    
    params[5] = Tobs-offset;
    params[6] = log(file_params[6]*PC_SI*1.0e6);
}

static int read_injection_file(const char *filename, double *file_params)
{
    FILE *in;
    
    in = fopen(filename, "r");
    if(in == NULL)
    {
        fprintf(stderr, "Could not open injection file %s\n", filename);
        return 1;
    }
    
    if(fscanf(in, "%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf",
              &file_params[0], &file_params[1], &file_params[2], &file_params[3], &file_params[4],
              &file_params[5], &file_params[6], &file_params[7], &file_params[8], &file_params[9],
              &file_params[10]) != NP)
    {
        fprintf(stderr, "Could not parse injection file %s\n", filename);
        fclose(in);
        return 1;
    }
    
    fclose(in);
    return 0;
}

static void write_injection_file(const char *filename, const double *file_params)
{
    FILE *out;
    
    out = fopen(filename, "w");
    if(out == NULL)
    {
        fprintf(stderr, "Could not open %s for writing\n", filename);
        exit(1);
    }
    
    fprintf(out, "%.16e %.16e %.16e %.16e %.16e %.16e %.16e %.16e %.16e %.16e %.16e\n",
            file_params[0], file_params[1], file_params[2], file_params[3], file_params[4],
            file_params[5], file_params[6], file_params[7], file_params[8], file_params[9],
            file_params[10]);
    fclose(out);
}

static double draw_bns_mass(gsl_rng *r)
{
    double m;
    
    do
    {
        m = BNS_MASS_MEAN_MSUN + BNS_MASS_SIGMA_MSUN*gsl_ran_gaussian(r, 1.0);
    }while(m <= 0.0 || m > BNS_MASS_MAX_MSUN);
    
    return m;
}

static void draw_bns_population(double *file_params, double Tobs, double offset, double max_luminosity_distance_mpc, gsl_rng *r)
{
    double zmax, dcmax, dc, dl, sindec;
    
    if(max_luminosity_distance_mpc <= 0.0)
    {
        fprintf(stderr, "Invalid BNS population maximum luminosity distance: %e Mpc\n", max_luminosity_distance_mpc);
        exit(1);
    }
    
    zmax = lcdm_redshift_from_luminosity_distance(max_luminosity_distance_mpc);
    dcmax = lcdm_comoving_distance_mpc(zmax);
    dc = dcmax*cbrt(gsl_rng_uniform_pos(r));
    dl = lcdm_luminosity_distance_from_comoving_distance(dc);
    
    file_params[0] = draw_bns_mass(r);
    file_params[1] = draw_bns_mass(r);
    file_params[2] = BNS_SPIN_MAX*gsl_rng_uniform(r);
    file_params[3] = BNS_SPIN_MAX*gsl_rng_uniform(r);
    file_params[4] = TPI*gsl_rng_uniform(r);
    file_params[5] = Tobs-offset;
    file_params[6] = dl;
    file_params[7] = TPI*gsl_rng_uniform(r);
    sindec = -1.0 + 2.0*gsl_rng_uniform(r);
    file_params[8] = asin(sindec);
    file_params[9] = PI*gsl_rng_uniform(r);
    file_params[10] = -1.0 + 2.0*gsl_rng_uniform(r);
}

static double lcdm_eofz(double z)
{
    return sqrt(LCDM_OMEGA_M*pow(1.0+z, 3.0) + LCDM_OMEGA_L);
}

static double lcdm_comoving_distance_mpc(double z)
{
    int i, n;
    double h, sum, zi;
    
    if(z <= 0.0) return 0.0;
    
    n = 1024;
    h = z/(double)n;
    sum = 1.0/lcdm_eofz(0.0) + 1.0/lcdm_eofz(z);
    
    for(i = 1; i < n; i++)
    {
        zi = h*(double)i;
        sum += ((i%2 == 0) ? 2.0 : 4.0)/lcdm_eofz(zi);
    }
    
    return (CLIGHT/1000.0)/LCDM_H0_KM_S_MPC*h*sum/3.0;
}

static double lcdm_luminosity_distance_from_comoving_distance(double dc_mpc)
{
    double z;
    
    z = lcdm_redshift_from_comoving_distance(dc_mpc);
    return (1.0+z)*dc_mpc;
}

static double lcdm_redshift_from_comoving_distance(double dc_mpc)
{
    int i;
    double zlow, zhigh, zmid;
    
    if(dc_mpc <= 0.0) return 0.0;
    
    zlow = 0.0;
    zhigh = 1.0;
    while(lcdm_comoving_distance_mpc(zhigh) < dc_mpc) zhigh *= 2.0;
    
    for(i = 0; i < 80; i++)
    {
        zmid = 0.5*(zlow+zhigh);
        if(lcdm_comoving_distance_mpc(zmid) < dc_mpc) zlow = zmid;
        else zhigh = zmid;
    }
    
    return 0.5*(zlow+zhigh);
}

static double lcdm_redshift_from_luminosity_distance(double dl_mpc)
{
    int i;
    double zlow, zhigh, zmid, dlmid;
    
    if(dl_mpc <= 0.0) return 0.0;
    
    zlow = 0.0;
    zhigh = 1.0;
    while((1.0+zhigh)*lcdm_comoving_distance_mpc(zhigh) < dl_mpc) zhigh *= 2.0;
    
    for(i = 0; i < 80; i++)
    {
        zmid = 0.5*(zlow+zhigh);
        dlmid = (1.0+zmid)*lcdm_comoving_distance_mpc(zmid);
        if(dlmid < dl_mpc) zlow = zmid;
        else zhigh = zmid;
    }
    
    return 0.5*(zlow+zhigh);
}

static double compute_detection_snr(struct Net *net, Detector *network, double **D, double **SN, double *params, RealVector *freq, int N, double Tobs, double *detector_snr)
{
#if USE_FAST_DETECTION_SNR
    double fast_snr2, exact_snr, exact_snr2, rel;
    
    fast_snr2 = fast_network_snr2(net, network, SN, params, Tobs, N, detector_snr);
    
#if FAST_SNR_EXACT_CHECK
    exact_snr = compute_exact_detection_snr(net, network, D, SN, params, freq, N, Tobs, NULL);
    exact_snr2 = exact_snr*exact_snr;
    rel = (exact_snr2 > 0.0) ? (fast_snr2-exact_snr2)/exact_snr2 : 0.0;
    printf("Fast SNR^2 check: fast=%e exact=%e relerr=%e\n", fast_snr2, exact_snr2, rel);
#endif
    
    return sqrt(fast_snr2);
#else
    return compute_exact_detection_snr(net, network, D, SN, params, freq, N, Tobs, detector_snr);
#endif
}

static double compute_exact_detection_snr(struct Net *net, Detector *network, double **D, double **SN, double *params, RealVector *freq, int N, double Tobs, double *detector_snr)
{
    int id;
    double logL, snr2;
    
    fulltemplates(net, network, D, freq, params, N);
    
    for(id = 0; id < net->Nifo; ++id)
    {
        snr2 = fourier_nwip(D[id], D[id], SN[id], 1, N/2, N);
        if(detector_snr != NULL) detector_snr[id] = sqrt(snr2);
    }
    
    logL = log_likelihood_full(net, network, D, SN, params, freq, N, Tobs);
    if(logL <= 0.0) return 0.0;
    
    return sqrt(2.0*logL);
}

static double fast_network_snr2(struct Net *net, Detector *network, double **SN, double *params, double Tobs, int N, double *detector_snr)
{
    int id, ifo;
    double alpha, delta, psi, ciota, Ap, Ac, Fs2;
    double det_snr2, network_snr2;
    double detector_norm2[DETECTOR_CATALOG_SIZE];
    double Fp[DETECTOR_CATALOG_SIZE], Fc[DETECTOR_CATALOG_SIZE], dtimes[DETECTOR_CATALOG_SIZE];
    
    if(fast_detector_snr2_norms(net, SN, params, Tobs, N, detector_norm2) != 0)
    {
        if(detector_snr != NULL)
        {
            for(id = 0; id < net->Nifo; ++id) detector_snr[id] = 0.0;
        }
        return 0.0;
    }
    
    alpha = params[7];
    delta = params[8];
    psi = params[9];
    ciota = params[10];
    Ap = 0.5*(1.0+ciota*ciota);
    Ac = -ciota;
    
    Response(network, alpha, delta, psi, net->GMST, Fp, Fc, dtimes);
    
    network_snr2 = 0.0;
    for(id = 0; id < net->Nifo; ++id)
    {
        ifo = net->labels[id];
        Fs2 = Ap*Ap*Fp[ifo]*Fp[ifo] + Ac*Ac*Fc[ifo]*Fc[ifo];
        det_snr2 = Fs2*detector_norm2[id];
        if(det_snr2 < 0.0) det_snr2 = 0.0;
        if(detector_snr != NULL) detector_snr[id] = sqrt(det_snr2);
        network_snr2 += det_snr2;
    }
    
    return network_snr2;
}

static int fast_detector_snr2_norms(struct Net *net, double **SN, double *params, double Tobs, int N, double *detector_norm2)
{
    int i, id, nfast;
    double fs, fmin, fmax, log_fmin, log_fmax;
    double m1_SI, m2_SI, chi1, chi2, distance, phi0;
    double sn, det_snr2;
    double *fgrid, *base, *integrand;
    RealVector *freq_fast;
    AmpPhaseFDWaveform *ap = NULL;
    gsl_interp_accel *acc;
    gsl_spline *spline;
    
    for(id = 0; id < net->Nifo; ++id) detector_norm2[id] = 0.0;
    
    nfast = FAST_SNR_GRID_SIZE;
    if(nfast < 5) nfast = 5;
    
    fs = fbegin(params);
    fmin = fs;
    if(fmin < 1.0/Tobs) fmin = 1.0/Tobs;
    fmax = ((double)(N/2-1))/Tobs;
    
    if(fmin >= fmax)
    {
        return 0;
    }
    
    fgrid = (double *)malloc(sizeof(double)*nfast);
    base = (double *)malloc(sizeof(double)*nfast);
    integrand = (double *)malloc(sizeof(double)*nfast);
    freq_fast = CreateRealVector(nfast);
    
    if(fgrid == NULL || base == NULL || integrand == NULL || freq_fast == NULL)
    {
        fprintf(stderr, "Could not allocate fast SNR workspace\n");
        free(fgrid);
        free(base);
        free(integrand);
        if(freq_fast != NULL) DestroyRealVector(freq_fast);
        return 1;
    }
    
    log_fmin = log(fmin);
    log_fmax = log(fmax);
    for(i = 0; i < nfast; ++i)
    {
        fgrid[i] = exp(log_fmin + (log_fmax-log_fmin)*(double)i/(double)(nfast-1));
        freq_fast->data[i] = fgrid[i];
    }
    
    m1_SI = params[0]*MSUN_SI;
    m2_SI = params[1]*MSUN_SI;
    chi1 = params[2];
    chi2 = params[3];
    phi0 = 0.5*params[4];
    distance = exp(params[6]);
    
    IMRPhenomDGenerateh22FDAmpPhase(&ap, freq_fast, 0, phi0, fref, m1_SI, m2_SI, chi1, chi2, distance);
    if(ap == NULL)
    {
        fprintf(stderr, "Could not generate waveform for fast SNR calculation\n");
        free(fgrid);
        free(base);
        free(integrand);
        DestroyRealVector(freq_fast);
        return 1;
    }
    
    for(i = 0; i < nfast; ++i)
    {
        base[i] = 4.0*h22fac*h22fac*ap->amp[i]*ap->amp[i];
    }
    
    acc = gsl_interp_accel_alloc();
    spline = gsl_spline_alloc(gsl_interp_akima, nfast);
    if(acc == NULL || spline == NULL)
    {
        fprintf(stderr, "Could not allocate GSL spline workspace for fast SNR calculation\n");
        if(spline != NULL) gsl_spline_free(spline);
        if(acc != NULL) gsl_interp_accel_free(acc);
        DestroyAmpPhaseFDWaveform(ap);
        DestroyRealVector(freq_fast);
        free(fgrid);
        free(base);
        free(integrand);
        return 1;
    }
    
    for(id = 0; id < net->Nifo; ++id)
    {
        for(i = 0; i < nfast; ++i)
        {
            sn = interpolate_uniform_spectrum(SN[id], N/2, Tobs, fgrid[i]);
            integrand[i] = (sn > 0.0) ? base[i]/sn : 0.0;
        }
        
        gsl_spline_init(spline, fgrid, integrand, nfast);
        det_snr2 = gsl_spline_eval_integ(spline, fgrid[0], fgrid[nfast-1], acc);
        if(det_snr2 < 0.0) det_snr2 = 0.0;
        detector_norm2[id] = det_snr2;
        gsl_interp_accel_reset(acc);
    }
    
    gsl_spline_free(spline);
    gsl_interp_accel_free(acc);
    DestroyAmpPhaseFDWaveform(ap);
    DestroyRealVector(freq_fast);
    free(fgrid);
    free(base);
    free(integrand);
    
    return 0;
}

/*
 * Estimate the network BNS range using the standard LIGO convention.
 *
 * The BNS range is not the arithmetic mean of the distance at which a source
 * crosses threshold. It is the radius of a sphere whose volume equals the
 * orientation-averaged sensitive volume,
 *
 *     D_range = <D_thr^3>^(1/3).
 *
 * For this diagnostic the source is a non-spinning 1.4+1.4 Msun binary. We
 * sample the extrinsic angles uniformly in the usual physical variables:
 * right ascension in [0, 2*pi), sin(declination) in [-1, 1], polarization in
 * [0, pi), and cos(inclination) in [-1, 1].
 *
 * There is no need to Monte Carlo over distance. We compute the network SNR at
 * a fixed reference luminosity distance, then use rho proportional to 1/D:
 *
 *     D_thr = D_ref * rho(D_ref) / rho_threshold.
 *
 * The expensive part of the SNR integral is independent of sky position and
 * orientation for fixed masses, spins, and distance, so fast_detector_snr2_norms()
 * precomputes one coarse-grid integral per detector. Each Monte Carlo draw then
 * only needs the antenna/orientation factor
 *
 *     A_plus^2 F_plus^2 + A_cross^2 F_cross^2.
 *
 * The mean_threshold_distance output is a diagnostic. The max_threshold_distance
 * output is the largest threshold distance encountered in the Monte Carlo.
 * Catalog mode multiplies this by BNS_POPULATION_DISTANCE_SCALE before using it
 * as the maximum luminosity distance for rejected BNS population draws. This
 * keeps the population cutoff tied to the selected detector network and PSDs,
 * while still allowing for louder, higher-chirp-mass catalog sources.
 */
static double estimate_bns_range(struct Net *net, Detector *network, double **SN, int N, double Tobs, gsl_rng *r, double *mean_threshold_distance, double *max_threshold_distance)
{
    int i, id, ifo;
    double file_params[NP], params[NP];
    double detector_norm2[DETECTOR_CATALOG_SIZE];
    double Fp[DETECTOR_CATALOG_SIZE], Fc[DETECTOR_CATALOG_SIZE], dtimes[DETECTOR_CATALOG_SIZE];
    double alpha, delta, sindec, psi, ciota, Ap, Ac, Fs2;
    double snr2_ref, dthr, sum_d, sum_d3, max_d;
    
    if(BNS_RANGE_MC_DRAWS < 1)
    {
        fprintf(stderr, "BNS_RANGE_MC_DRAWS must be positive\n");
        return -1.0;
    }
    
    for(i = 0; i < NP; ++i) file_params[i] = 0.0;
    file_params[0] = BNS_RANGE_COMPONENT_MASS_MSUN;
    file_params[1] = BNS_RANGE_COMPONENT_MASS_MSUN;
    file_params[2] = 0.0;
    file_params[3] = 0.0;
    file_params[4] = 0.0;
    file_params[5] = Tobs-net->offset;
    file_params[6] = BNS_RANGE_REFERENCE_DISTANCE_MPC;
    file_params[7] = 0.0;
    file_params[8] = 0.0;
    file_params[9] = 0.0;
    file_params[10] = 1.0;
    prepare_internal_params(params, file_params, Tobs, net->offset);
    
    if(fast_detector_snr2_norms(net, SN, params, Tobs, N, detector_norm2) != 0) return -1.0;
    
    sum_d = 0.0;
    sum_d3 = 0.0;
    max_d = 0.0;
    
    for(i = 0; i < BNS_RANGE_MC_DRAWS; ++i)
    {
        alpha = TPI*gsl_rng_uniform(r);
        sindec = -1.0 + 2.0*gsl_rng_uniform(r);
        delta = asin(sindec);
        psi = PI*gsl_rng_uniform(r);
        ciota = -1.0 + 2.0*gsl_rng_uniform(r);
        Ap = 0.5*(1.0+ciota*ciota);
        Ac = -ciota;
        
        Response(network, alpha, delta, psi, net->GMST, Fp, Fc, dtimes);
        
        snr2_ref = 0.0;
        for(id = 0; id < net->Nifo; ++id)
        {
            ifo = net->labels[id];
            Fs2 = Ap*Ap*Fp[ifo]*Fp[ifo] + Ac*Ac*Fc[ifo]*Fc[ifo];
            snr2_ref += Fs2*detector_norm2[id];
        }
        
        if(snr2_ref < 0.0) snr2_ref = 0.0;
        dthr = BNS_RANGE_REFERENCE_DISTANCE_MPC*sqrt(snr2_ref)/BNS_SNR_THRESHOLD;
        
        sum_d += dthr;
        sum_d3 += dthr*dthr*dthr;
        if(dthr > max_d) max_d = dthr;
    }
    
    if(mean_threshold_distance != NULL) *mean_threshold_distance = sum_d/(double)BNS_RANGE_MC_DRAWS;
    if(max_threshold_distance != NULL) *max_threshold_distance = max_d;
    
    // LIGO range is the radius of the sphere with the same orientation-averaged sensitive volume.
    return cbrt(sum_d3/(double)BNS_RANGE_MC_DRAWS);
}

static double interpolate_uniform_spectrum(double *y, int n, double Tobs, double f)
{
    int i;
    double x, frac;
    
    if(f <= 0.0) return y[0];
    
    x = f*Tobs;
    i = (int)floor(x);
    
    if(i < 0) return y[0];
    if(i >= n-1) return y[n-1];
    
    frac = x-(double)i;
    return y[i]*(1.0-frac) + y[i+1]*frac;
}

static void setup_source_likelihood(struct Net *net, Detector *network, double *params, double **D, double **SN, double **data, double *hwave, double ***wave, double *DD, double **WW, double ***DHc, double ***DHs, double ***HH, double **skyx, double **paramx, double **pallx, int *who, double *heat, int *mxc, double *logLsky, RealVector *freq, int N, int nt, double Tobs)
{
    int i, j, id;
    double x;
    
    initialize_chain_state(net, params, paramx, pallx, skyx, who, heat, mxc);
    fulltemplates(net, network, D, freq, params, N);
    
    for(id = 0; id < net->Nifo; ++id)
    {
        data[id][0] = 0.0;
        data[id][N/2] = 0.0;
        for(i = 1; i < N/2; ++i)
        {
            x = 1.0/sqrt(SN[id][i]);
            data[id][i] = x*D[id][i];
            data[id][N-i] = x*D[id][N-i];
        }
    }
    
    geotemplate(hwave, freq, params, N);
    
    j = 1;
    for(id = 0; id < net->Nifo; ++id)
    {
        wave[j][id][0] = 0.0;
        wave[j][id][N/2] = 0.0;
        for(i = 1; i < N/2; ++i)
        {
            x = 1.0/sqrt(SN[id][i]);
            wave[j][id][i] = hwave[i]*x;
            wave[j][id][N-i] = hwave[N-i]*x;
        }
    }
    
    for(j = 2; j <= NC; j++)
    {
        for(id = 0; id < net->Nifo; ++id)
        {
            for(i = 1; i < N/2; ++i)
            {
                wave[j][id][i] = wave[1][id][i];
                wave[j][id][N-i] = wave[1][id][N-i];
            }
        }
    }
    
    j = 1;
    skylikesetup(net, data, wave[j], DD, WW[j], DHc[j], DHs[j], Tobs, N, nt);
    fisherskysetup(net, wave[j], HH[j], Tobs, N);
    
    for(j = 2; j <= NC; j++)
    {
        for(id = 0; id < net->Nifo; ++id)
        {
            WW[j][id] = WW[1][id];
            for(i = 0; i < nt; ++i)
            {
                DHc[j][id][i] = DHc[1][id][i];
                DHs[j][id][i] = DHs[1][id][i];
            }
            for(i = 0; i < 3; ++i)
            {
                HH[j][id][i] = HH[1][id][i];
            }
        }
    }
    
    logLsky[1] = skylike(net, network, skyx[1], DD, WW[1], DHc[1], DHs[1], Tobs/(double)N, nt, 0);
    //printf("%f\n", logLsky[1]);
}

static int run_source_mcmc(struct Net *net, Detector *network, double *params, double **D, double **SN, double **data, double *hwave, double ***wave, double *DD, double **WW, double ***DHc, double ***DHs, double ***HH, double **skyx, double **paramx, double **pallx, int *who, double *heat, int *mxc, double *logLsky, RealVector *freq, int N, int nt, double Tobs, double dt, int Nsky, gsl_rng *r, SkyPostSummary *summary)
{
    FILE *chain;
    
    setup_source_likelihood(net, network, params, D, SN, data, hwave, wave, DD, WW, DHc, DHs, HH, skyx, paramx, pallx, who, heat, mxc, logLsky, freq, N, nt, Tobs);
    
    chain = fopen("skychain.dat", "w");
    if(chain == NULL)
    {
        fprintf(stderr, "Could not open skychain.dat for writing\n");
        return 1;
    }
    
    skymcmc(net, network, Nsky, mxc, chain, paramx, skyx, pallx, who, heat, dt, nt, DD, WW, DHc, DHs, HH, Tobs, r);
    fclose(chain);
    
    if(postprocess_sky_chain("skychain.dat", SKYMAP_NSIDE, summary) != 0)
    {
        fprintf(stderr, "Sky-chain post-processing failed\n");
        return 1;
    }
    
    return 0;
}

static int ensure_directory(const char *path)
{
    struct stat st;
    
    if(stat(path, &st) == 0)
    {
        if(S_ISDIR(st.st_mode)) return 0;
        fprintf(stderr, "%s exists but is not a directory\n", path);
        return 1;
    }
    
    if(mkdir(path, 0775) != 0 && errno != EEXIST)
    {
        fprintf(stderr, "Could not create directory %s\n", path);
        return 1;
    }
    
    return 0;
}

static int move_file_to_source_name(const char *source, const char *dest_dir, const char *prefix, int source_id)
{
    char dest[1024];
    
    snprintf(dest, sizeof(dest), "%s/%s_%d.dat", dest_dir, prefix, source_id);
    if(rename(source, dest) != 0)
    {
        fprintf(stderr, "Could not move %s to %s\n", source, dest);
        return 1;
    }
    
    return 0;
}

static int write_summary_file(const char *filename, int source_id, int attempts, const SkyPostSummary *summary, struct Net *net, Detector *network, const double *detector_snr, double network_snr)
{
    int id, ifo;
    FILE *out;
    
    out = fopen(filename, "w");
    if(out == NULL)
    {
        fprintf(stderr, "Could not open %s for writing\n", filename);
        return 1;
    }
    
    fprintf(out, "# source_id attempts area50_sq_deg area90_sq_deg boundary_pixels boundary_area_sq_deg map_theta_deg map_phi_deg network_snr\n");
    fprintf(out, "%d %d %.16e %.16e %ld %.16e %.16e %.16e %.16e\n",
            source_id, attempts, summary->area50, summary->area90, summary->boundary_pixels,
            summary->boundary_area, summary->map_theta, summary->map_phi, network_snr);
    fprintf(out, "# detector_index detector_label detector_name snr\n");
    for(id = 0; id < net->Nifo; ++id)
    {
        ifo = net->labels[id];
        fprintf(out, "%d %d \"%s\" %.16e\n", id, ifo, network[ifo].name, detector_snr[id]);
    }
    
    fclose(out);
    return 0;
}

static int write_population_histograms(const char *dest_dir, const double *area50, const double *area90, const double *distance, const double *snr, int n, double distance_max_mpc)
{
    char filename[1024];
    double area50_max, area90_max, distance_max, snr_max;
    
    if(n < 1) return 0;
    
    area50_max = finite_array_max(area50, n);
    area90_max = finite_array_max(area90, n);
    distance_max = distance_max_mpc;
    if(!isfinite(distance_max) || distance_max <= 0.0) distance_max = finite_array_max(distance, n);
    snr_max = finite_array_max(snr, n);
    
    snprintf(filename, sizeof(filename), "%s/area50_histogram.dat", dest_dir);
    if(write_histogram_file(filename, "area50", "square_degrees", area50, n, POPULATION_HISTOGRAM_BINS, 0.0, area50_max) != 0) return 1;
    
    snprintf(filename, sizeof(filename), "%s/area90_histogram.dat", dest_dir);
    if(write_histogram_file(filename, "area90", "square_degrees", area90, n, POPULATION_HISTOGRAM_BINS, 0.0, area90_max) != 0) return 1;
    
    snprintf(filename, sizeof(filename), "%s/distance_histogram.dat", dest_dir);
    if(write_histogram_file(filename, "luminosity_distance", "Mpc", distance, n, POPULATION_HISTOGRAM_BINS, 0.0, distance_max) != 0) return 1;
    
    snprintf(filename, sizeof(filename), "%s/snr_histogram.dat", dest_dir);
    if(write_histogram_file(filename, "network_snr", "dimensionless", snr, n, POPULATION_HISTOGRAM_BINS, BNS_SNR_THRESHOLD, snr_max) != 0) return 1;
    
    printf("Population histograms written to %s/area50_histogram.dat, %s/area90_histogram.dat, %s/distance_histogram.dat, and %s/snr_histogram.dat\n",
           dest_dir, dest_dir, dest_dir, dest_dir);
    
    return 0;
}

static int write_histogram_file(const char *filename, const char *quantity, const char *units, const double *values, int n, int nbins, double xmin, double xmax)
{
    int i, bin, valid;
    long *counts, underflow, overflow;
    double x, width, center, fraction, density;
    double sample_min, sample_max, sample_sum;
    FILE *out;
    
    if(nbins < 1) nbins = 1;
    if(!isfinite(xmin)) xmin = 0.0;
    if(!isfinite(xmax) || xmax <= xmin) xmax = finite_array_max(values, n);
    if(!isfinite(xmax) || xmax <= xmin) xmax = xmin + 1.0;
    
    counts = (long *)calloc((size_t)nbins, sizeof(long));
    if(counts == NULL)
    {
        fprintf(stderr, "Could not allocate histogram counts for %s\n", filename);
        return 1;
    }
    
    width = (xmax-xmin)/(double)nbins;
    valid = 0;
    underflow = 0;
    overflow = 0;
    sample_min = 0.0;
    sample_max = 0.0;
    sample_sum = 0.0;
    
    for(i = 0; i < n; ++i)
    {
        x = values[i];
        if(!isfinite(x)) continue;
        
        if(valid == 0)
        {
            sample_min = x;
            sample_max = x;
        }
        else
        {
            if(x < sample_min) sample_min = x;
            if(x > sample_max) sample_max = x;
        }
        sample_sum += x;
        valid++;
        
        if(x < xmin)
        {
            bin = 0;
            underflow++;
        }
        else if(x > xmax)
        {
            bin = nbins-1;
            overflow++;
        }
        else
        {
            bin = (int)floor((x-xmin)/width);
            if(bin < 0) bin = 0;
            if(bin >= nbins) bin = nbins-1;
        }
        counts[bin]++;
    }
    
    if(valid == 0)
    {
        fprintf(stderr, "No finite values available for histogram %s\n", filename);
        free(counts);
        return 1;
    }
    
    out = fopen(filename, "w");
    if(out == NULL)
    {
        fprintf(stderr, "Could not open %s for writing\n", filename);
        free(counts);
        return 1;
    }
    
    fprintf(out, "# quantity %s\n", quantity);
    fprintf(out, "# units %s\n", units);
    fprintf(out, "# samples %d finite_samples %d\n", n, valid);
    fprintf(out, "# bins %d\n", nbins);
    fprintf(out, "# histogram_range %.16e %.16e\n", xmin, xmax);
    fprintf(out, "# sample_min_max_mean %.16e %.16e %.16e\n", sample_min, sample_max, sample_sum/(double)valid);
    fprintf(out, "# underflow_overflow %ld %ld\n", underflow, overflow);
    fprintf(out, "# columns bin_low bin_high bin_center count fraction density\n");
    
    for(i = 0; i < nbins; ++i)
    {
        x = xmin + width*(double)i;
        center = x + 0.5*width;
        fraction = (double)counts[i]/(double)valid;
        density = fraction/width;
        fprintf(out, "%.16e %.16e %.16e %ld %.16e %.16e\n",
                x, x+width, center, counts[i], fraction, density);
    }
    
    fclose(out);
    free(counts);
    
    return 0;
}

static double finite_array_max(const double *values, int n)
{
    int i, found;
    double xmax;
    
    xmax = 0.0;
    found = 0;
    
    for(i = 0; i < n; ++i)
    {
        if(!isfinite(values[i])) continue;
        if(!found || values[i] > xmax)
        {
            xmax = values[i];
            found = 1;
        }
    }
    
    return found ? xmax : 0.0;
}

void load_psd(const char *filename, double *Sn, double *freqs, int nfreq, double fmin, double fmax)
{
    int i, M;
    double f, a, fstart, fend;
    double *freq_read, *asd_read;
    FILE *in;
    gsl_spline *spline;
    gsl_interp_accel *acc;
    
    in = fopen(filename, "r");
    if(in == NULL)
    {
        fprintf(stderr, "Could not open PSD file %s\n", filename);
        exit(1);
    }
    
    M = 0;
    while(fscanf(in, "%lf%lf", &f, &a) == 2) M++;
    
    if(M < 5)
    {
        fprintf(stderr, "PSD file %s has too few samples for Akima interpolation\n", filename);
        fclose(in);
        exit(1);
    }
    
    rewind(in);
    
    freq_read = double_vector(M);
    asd_read = double_vector(M);
    
    for(i = 0; i < M; ++i)
    {
        if(fscanf(in, "%lf%lf", &freq_read[i], &asd_read[i]) != 2)
        {
            fprintf(stderr, "Could not parse PSD file %s\n", filename);
            fclose(in);
            exit(1);
        }
    }
    fclose(in);
    
    for(i = 1; i < M; ++i)
    {
        if(freq_read[i] <= freq_read[i-1])
        {
            fprintf(stderr, "PSD file %s frequencies must be strictly increasing\n", filename);
            exit(1);
        }
    }
    
    fstart = freq_read[0];
    fend = freq_read[M-1];
    
    acc = gsl_interp_accel_alloc();
    spline = gsl_spline_alloc(gsl_interp_akima, M);
    gsl_spline_init(spline, freq_read, asd_read, M);
    
    for(i = 0; i < nfreq; ++i)
    {
        Sn[i] = 1.0;
        if(freqs[i] >= fmin && freqs[i] <= fmax && freqs[i] >= fstart && freqs[i] <= fend)
        {
            a = gsl_spline_eval(spline, freqs[i], acc);
            Sn[i] = a*a;
        }
    }
    
    gsl_spline_free(spline);
    gsl_interp_accel_free(acc);
    free_double_vector(freq_read);
    free_double_vector(asd_read);
}

void skymcmc(struct Net *net, Detector *network, int MCX, int *mxc, FILE *chain, double **paramx, double **skyx, double **pallx, int *who, double *heat, double dtx, int nt, double *DD, double **WW, double ***DHc,  double ***DHs, double ***HH, double Tobs, gsl_rng * r)
{
    int i, j, k, q, ic, id1, id2;
    int map_ok;
    int scount, sacc, hold, mcount;
    int ac, rc, rca, apc, apa, clc, cla, PRcnt, POcnt;
    int sdx, Ax, mc;
    double alpha, beta;
    double Mchirp, Mtot, eta, dm, m1, m2, chieff, ciota;
    double qxy, qyx, Jack, phi;
    int rflag, aflag, cflag;
    double **sky, **skyy, **skyh;
    double x, y, z, DL, scale, logLy;
    double *logLx;
    double pAy, pSy, logH, pAx, pSx;
    double *param;
    double ***fishskyx, ***fishskyy;
    double ***skyvecsx, ***skyvecsy;
    double **skyevalsx, **skyevalsy;
    double Fp, Fc, Fs, ps;
    double *jump, *sqH;
    double ldetx, ldety;
    double scmax, scmin;
    double DLx, DLy, cosdeltax, cosdeltay;
    int fflag, fc, fac;
    int uflag, uc, uac;
    double Ap, Ac, Fcross, Fplus, lambda, lambda2, Fs2;
    double delta, psi;
    double Fpa[DETECTOR_CATALOG_SIZE], Fca[DETECTOR_CATALOG_SIZE], delays[DETECTOR_CATALOG_SIZE];
    double Fpa2[DETECTOR_CATALOG_SIZE], Fca2[DETECTOR_CATALOG_SIZE], delays2[DETECTOR_CATALOG_SIZE];
    
    param = (double*)malloc(sizeof(double)*(NX+3*net->Nifo));
    
    // sky parameter order
    //[0] alpha, [1] delta [2] psi [3] ciota [4] scale [5] phi0 [6] dt
    

    
    // max and min of rescaling parameter
    scmin = 0.1;
    scmax = 10.0;
    
    sky = double_matrix(NC+1,NS);
    skyh = double_matrix(NC+1,NS);
    skyy = double_matrix(NC+1,NS);
    logLx = double_vector(NC+1);
    sqH = double_vector(NC+1);
    
    for(k = 1; k <= NC; k++)
    {
        for(i = 0; i < NS; i++) sky[k][i] = skyx[k][i];
        for(i = 0; i < NS; i++) skyh[k][i] = skyx[k][i];
    }
    
    for(k = 1; k <= NC; k++) sqH[k] = sqrt(heat[k]);
    
    ic = who[1];
    
    for(k = 1; k <= NC; k++) logLx[k] = skylike(net, network, skyx[k], DD, WW[k], DHc[k], DHs[k], dtx, nt, 0);
    
    
    fishskyx = double_tensor(NC+1,NS,NS);
    fishskyy = double_tensor(NC+1,NS,NS);
    skyvecsx = double_tensor(NC+1,NS,NS);
    skyvecsy = double_tensor(NC+1,NS,NS);
    skyevalsx = double_matrix(NC+1,NS);
    skyevalsy = double_matrix(NC+1,NS);
    jump = double_vector(NS);
    
    
    for(k = 1; k <= NC; k++)
    {
    fisher_matrix_fastsky(net, network, skyx[k], fishskyx[k], HH[k]);
    FisherEvec(fishskyx[k], skyevalsx[k], skyvecsx[k], NS);
    }
    
    
    //printf("Extrinsic MCMC\n");
    
    
    ac = 0;
    rc = 1;
    rca = 0;
    apc = 0;
    apa = 0;
    fc = 1;
    uc = 1;
    
    clc = 0;
    cla = 0;
    fac = 0;
    uac = 0;
    
    PRcnt = 0;
    POcnt = 0;
    
    sdx = 0.0;
    Ax = 0.0;
    
    scount = 0;
    sacc = 0;
    mcount = 0;
    
    for(mc = 0; mc < MCX; mc++)
    {
        
        
        if(mc > 1 && mc%1000==0)
        {
            // update the Fisher matrices
            for(k = 1; k <= NC; k++)
            {
                fisher_matrix_fastsky(net, network, skyx[k], fishskyx[k], HH[k]);
                FisherEvec(fishskyx[k], skyevalsx[k], skyvecsx[k], NS);
            }
        }
        
        alpha = gsl_rng_uniform(r);
        
        if((NC > 1) && (alpha < 0.2))  // decide if we are doing a MCMC update of all the chains or a PT swap
        {
            
            // chain swap
            scount++;
            
            alpha = (double)(NC-1)*gsl_rng_uniform(r);
            j = (int)(alpha) + 1;
            beta = exp((logLx[who[j]]-logLx[who[j+1]])/heat[j+1] - (logLx[who[j]]-logLx[who[j+1]])/heat[j]);
            alpha = gsl_rng_uniform(r);
            if(beta > alpha)
            {
                hold = who[j];
                who[j] = who[j+1];
                who[j+1] = hold;
                sacc++;
            }
            
            
            
        }
        else      // MCMC update
        {
            
            mcount++;

            
            for(k = 1; k <= NC; k++)
            {
                for(i = 0; i < NS; i++) skyy[k][i] = skyx[k][i];
            }
            
            for(k=1; k <= NC; k++)
            {
                q = who[k];
                
                qxy = 0.0;
                qyx = 0.0;
                
                Jack = 0.0;
                map_ok = 1;
                
                rflag = 0;
                aflag = 0;
                cflag = 0;
                fflag = 0;
                uflag = 0;
                
                
                alpha = gsl_rng_uniform(r);
                
                if(alpha > 0.85 && net->Nifo > 1)  // ring
                {
                    

                    id1 = 0;
                    id2 = 1;
                    
                    // Pick a pair of interferometers to define sky ring
                    if(net->Nifo > 2)
                    {
                      id1 = (int)((double)(net->Nifo)*gsl_rng_uniform(r));
                      do
                      {
                       id2 = (int)((double)(net->Nifo)*gsl_rng_uniform(r));
                      }while(id1==id2);
                    }
                    
                    // map these labels to actual detectors
                    id1 = net->labels[id1];
                    id2 = net->labels[id2];
                    
                    Ring(network, skyx[q], skyy[q], id1, id2, net->GMST, r);
                    qyx = 1.0;
                    qxy = 1.0;
                
                    
                    map_ok = skymap(network, skyx[q], skyy[q], net->GMST, id1, id2, net->labels[0]);
                    if(map_ok)
                    {
                        x = skydensity(network, skyx[q], skyy[q], net->GMST, id1, id2, net->labels[0]);
                        if(x > 0.0 && isfinite(x)) Jack = log(x);
                        else map_ok = 0;
                    }
                    
                    // The mapping only overs half the phi, psi space. Can cover it all by radomly shifting both by a half period
                    x = gsl_rng_uniform(r);
                    if(x > 0.5)
                    {
                        skyy[q][5] += PI;
                        skyy[q][2] += PI/2.0;
                    }
                    
                    if(k==1) rc++;
                    
                    rflag = 1;
                    
                }
                else if(alpha > 0.70 && net->Nifo > 2)  // detector-plane antipodal point
                {
                    
                    DetectorPlaneAntipode(network, skyx[q], skyy[q], net->labels[0], net->labels[1], net->labels[2], net->GMST);
                    qyx = 1.0;
                    qxy = 1.0;
                    
                    id1 = (int)((double)(net->Nifo)*gsl_rng_uniform(r));
                    do
                    {
                        id2 = (int)((double)(net->Nifo)*gsl_rng_uniform(r));
                    }while(id1==id2);
                    
                    id1 = net->labels[id1];
                    id2 = net->labels[id2];
                    
                    map_ok = skymap(network, skyx[q], skyy[q], net->GMST, id1, id2, net->labels[0]);
                    if(map_ok)
                    {
                        x = skydensity(network, skyx[q], skyy[q], net->GMST, id1, id2, net->labels[0]);
                        if(x > 0.0 && isfinite(x)) Jack = log(x);
                        else map_ok = 0;
                    }
                    
                    // Match the ring proposal's coverage of the phi, psi degeneracy.
                    x = gsl_rng_uniform(r);
                    if(x > 0.5)
                    {
                        skyy[q][5] += PI;
                        skyy[q][2] += PI/2.0;
                    }
                    
                    if(k==1) apc++;
                    
                    aflag = 1;
                    
                }
                else if (alpha > 0.10)  // Fisher matrix
                {
                    
                    if(k==1) fc++;
                    
                    fflag = 1;
                    
                    fisher_skyproposal(r, skyvecsx[q], skyevalsx[q], jump);
                    
                    for(i=0; i< NS; i++) skyy[q][i] = skyx[q][i]+sqH[k]*jump[i];
                    
                    // If the Fisher matrix was updated after each Fisher jump we would
                    // need these proposal densities. Since Fisher held fixed for blocks
                    // of iterations, we don't need the densities
                    // pfishxy = fisher_density(fishskyx, ldetx, skyx, skyy);
                    // fisher_matrix_fastsky(net, skyy, fishskyy, HH);
                    // fisher_skyvectors(fishskyy, skyvecsy, skyevalsy, &ldety);
                    //  pfishyx = fisher_density(fishskyy, ldety, skyy, skyx);
                }
                else  // jiggle (most useful early when Fisher not effective)
                {
                    
                    uflag = 1;
                    
                    if(k==1) uc++;
                    
                    beta = 0.01*pow(10.0, -floor(3.0*gsl_rng_uniform(r)))*sqH[k];
                    for(i = 0; i < NS-1; i++) skyy[q][i] = skyx[q][i]+beta*gsl_ran_gaussian(r,1.0);
                    skyy[q][6] = skyx[q][6]+0.01*beta*gsl_ran_gaussian(r,1.0);
                }
                
                
                if(skyy[q][0] > TPI) skyy[q][0] -= TPI;
                if(skyy[q][0] < 0.0) skyy[q][0] += TPI;
                if(skyy[q][2] > PI) skyy[q][2] -= PI;
                if(skyy[q][2] < 0.0) skyy[q][2] += PI;
                if(skyy[q][5] > TPI) skyy[q][5] -= TPI;
                if(skyy[q][5] < 0.0) skyy[q][5] += TPI;
                
                
                //[0] alpha, [1] delta [2] psi [3] cos(iota) [4] scale [5] phi0 [6] dt
                
 
                DLy = exp(pallx[q][6])/(skyy[q][4]*PC_SI);
                cosdeltax = cos(skyx[q][1]);
                cosdeltay = cos(skyy[q][1]);
                
                if(map_ok == 0 || DLy < DLmin || DLy > DLmax || fabs(skyy[q][1]) >= M_PI/2.0 || cosdeltax <= 0.0 || cosdeltay <= 0.0 || fabs(skyy[q][3]) > 1.0 || fabs(skyy[q][6]) > dtmax)
                {
                    logLy = -1.0e60;
                    
                    pAy = 0.0;
                    pAx = 0.0;
                    pSy = 0.0;
                    pSx = 0.0;
                    
                }
                else
                {
                    logLy = skylike(net, network, skyy[q], DD, WW[q], DHc[q], DHs[q], dtx, nt, 0);
                    
                    // Need a Jacobian a factor here since we sample uniformly in amplitude.
                    // Jacobian between D cos(theta) phi cos(iota) psi and A cos(theta) phi cos(iota) psi
                    // Since D = D_0/A, boils down to just D^2 |dD/dA| = D_0^3/A^4 = D^3/A
                
                    DLx = exp(pallx[q][6])/(skyx[q][4]);
                    pAx = 3.0*log(DLx)-log(skyx[q][4]);
  
                    DLy = exp(pallx[q][6])/(skyy[q][4]);
                    pAy = 3.0*log(DLy)- log(skyy[q][4]);
                    
                    // We sample in DEC but want a sky prior uniform in sin(DEC).
                    pSx = log(cosdeltax);
                    pSy = log(cosdeltay);
                    
                }
                
                
                logH = Jack + (logLy-logLx[q])/heat[k] + pAy+pSy-qyx-pAx-pSx+qxy;
                
                alpha = log(gsl_rng_uniform(r));
                
                
                if(logH > alpha)
                {
                    for(i=0; i< NS; i++) skyx[q][i] = skyy[q][i];
                    logLx[q] = logLy;
                    
                    if(k==1)
                    {
                        ac++;
                        if(rflag == 1) rca++;
                        if(aflag == 1) apa++;
                        if(fflag == 1) fac++;
                        if(uflag == 1) uac++;
                    }
                    
                }
                
            }  // ends loop over chains
            
        }  // ends choice of update
        
        /*
        if(mc%100 == 0)
        {
            ic = who[1];
            phi = skyx[ic][0];
            if(phi > TPI) phi -= TPI;
            if(phi < 0.0) phi += TPI;
            skyx[ic][0] = phi;
            Mchirp = exp(paramx[ic][0])/MSUN_SI;
            Mtot = exp(paramx[ic][1])/MSUN_SI;
            eta = pow((Mchirp/Mtot), (5.0/3.0));
            dm = sqrt(1.0-4.0*eta);
            m1 = Mtot*(1.0+dm)/2.0;
            m2 = Mtot*(1.0-dm)/2.0;
            chieff = (m1*paramx[ic][2]+m2*paramx[ic][3])/Mtot;

           
            // counter, log likelihood, chirp mass, total mass, effective spin, phase shift, time shift, distance, RA, sin(DEC),
            // polarization angle, cos inclination
            
            
            DL = exp(pallx[ic][6])/(1.0e6*PC_SI*skyx[ic][4]);
            z = z_DL(DL);
            
            // Note that skyx[ic][5], skyx[ic][6], hold different quantities than what is printed by the other MCMCs
 
            fprintf(chain,"%d %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e\n", mxc[1], logLx[ic], Mchirp, Mtot, chieff, skyx[ic][5], \
                    skyx[ic][6], DL, skyx[ic][0], sin(skyx[ic][1]), \
                    skyx[ic][2], skyx[ic][3], z, Mchirp/(1.0+z), Mtot/(1.0+z), m1/(1.0+z), m2/(1.0+z), m1, m2);
            
        
                    
            
             mxc[1] += 1;
            
          } */
        
#if PRINT_MCMC_STATE
        if(mc%10000 == 0)
        {
         ic = who[1];
         DL = exp(pallx[ic][6])/(1.0e6*PC_SI*skyx[ic][4]);
         printf("%d %f %f %f %f %f %f\n", mc, logLx[ic], DL, skyx[ic][0], skyx[ic][1], skyx[ic][2], skyx[ic][3]);
        }
#endif
        
        if(mc > MCX/4 && mc%10 == 0)
        {
         ic = who[1];
         DL = exp(pallx[ic][6])/(1.0e6*PC_SI*skyx[ic][4]);
         fprintf(chain, "%d %f %f %f %f\n", mc, logLx[ic], skyx[ic][0], sin(skyx[ic][1]), DL);
        }

    }
    
     // This is from the larger QuickCBC code that returns to full parameter updates, not just extrinsic
     // update the amplitude, time and phase shifts between detectors in preparation for extrinsic updates
     // for(k=1; k <= NC; k++) dshifts(net, network, skyx[k], paramx[k]);
    
    
    // sky  [0] alpha, [1] delta [2] psi [3] ciota [4] scale [5] dphi [6] dt
    // param [0] log(Mc) [1] log(Mt) [2] chi1 [3] chi2 [4] phi0  [5] tp0 [6] log(DL0) then relative amplitudes, time, phases

    // This is from the larger QuickCBC code that returns to full parameter u[dates, not just extrinsic
    // update the extrinsic parameters
    /* for(k = 1; k <= NC; k++)
    {
        
        // move reference point from geocenter to ref detector
        // Note that sky[4],sky[5], sky[6] hold shifts relative to the reference geocenter waveform
        // To map back to the reference detector
        
        ciota = skyh[k][3];
        Ap = (1.0+ciota*ciota)/2.0;
        Ac = -ciota;
        alpha = skyh[k][0];
        delta = skyh[k][1];
        psi = skyh[k][2];
        
        Response(network, alpha, delta, psi, net->GMST, Fpa, Fca, delays);
        Fplus = Fpa[net->labels[0]];
        Fcross = Fca[net->labels[0]];
        Fs = sqrt(Ap*Ap*Fplus*Fplus+Ac*Ac*Fcross*Fcross);
        lambda = atan2(Ac*Fcross,Ap*Fplus);
        if(lambda < 0.0) lambda += TPI;
        
        ciota = skyx[k][3];
        Ap = (1.0+ciota*ciota)/2.0;
        Ac = -ciota;
        alpha = skyx[k][0];
        delta = skyx[k][1];
        psi = skyx[k][2];
        
        Response(network, alpha, delta, psi, net->GMST, Fpa2, Fca2, delays2);
        Fplus = Fpa2[net->labels[0]];
        Fcross = Fca2[net->labels[0]];
        Fs2 = sqrt(Ap*Ap*Fplus*Fplus+Ac*Ac*Fcross*Fcross);
        lambda2 = atan2(Ac*Fcross,Ap*Fplus);
        if(lambda2 < 0.0) lambda2 += TPI;
        
        paramx[k][4] += 0.5*(skyx[k][5]+lambda2-lambda);
        while(paramx[k][4] > PI) paramx[k][4] -= PI;
        while(paramx[k][4] < 0.0) paramx[k][4] += PI;
        paramx[k][5] += skyx[k][6]+delays2[net->labels[0]]-delays[net->labels[0]];
        paramx[k][6] -= log(skyx[k][4]*Fs2/Fs);
        
        // sky will be re-aligned with geocenter so reset
        skyx[k][4] = 1.0;
        skyx[k][5] = 0.0;
        skyx[k][6] = 0.0;
       
    } */
    
    
    printf("Swap Acceptance = %f\n", (double)sacc/(double)(scount));
    printf("MCMC Acceptance = %f\n", (double)ac/(double)(mcount));
    printf("Ring Acceptance = %f\n", (double)rca/(double)(rc));
    printf("Antipodal Acceptance = %f\n", (apc > 0) ? (double)apa/(double)(apc) : 0.0);
    printf("Fisher Acceptance = %f\n", (double)fac/(double)(fc));
    printf("Jiggle Acceptance = %f\n", (double)uac/(double)(uc));
    
    
    
    free_double_matrix(skyy,NC+1);
    free_double_matrix(skyh,NC+1);
    free_double_matrix(sky,NC+1);
    free_double_vector(logLx);
    free_double_vector(sqH);
    free_double_tensor(fishskyx,NC+1,NS);
    free_double_tensor(fishskyy,NC+1,NS);
    free_double_tensor(skyvecsx,NC+1,NS);
    free_double_tensor(skyvecsy,NC+1,NS);
    free_double_matrix(skyevalsx,NC+1);
    free_double_matrix(skyevalsy,NC+1);
    free_double_vector(jump);
    
    free(param);
    
    return;
    
}

double log_likelihood_full(struct Net *net, Detector *network, double **D, double **SN, double *params, RealVector *freq, int N, double Tobs)
{
    int i;
    double logL;
    double **twave;
    double HH, HD, x;
    
    logL = 0.0;
        
        twave = double_matrix(net->Nifo,N);
    
        fulltemplates(net, network, twave, freq, params, N);
        
        logL = 0.0;
        for(i = 0; i < net->Nifo; i++)
        {
            HH = fourier_nwip(twave[i], twave[i], SN[i], 1, N/2, N);
            HD = fourier_nwip(D[i], twave[i], SN[i], 1, N/2, N);
            x = HD/sqrt(HH);
            //rho[i] = HD/sqrt(HH);
            logL += HD-0.5*HH;
            if(LOG_LIKELIHOOD_FULL_VERBOSE) printf("%d %f %f %f %f\n", i, HH, HD, sqrt(HH), x);
        }
        
        free_double_matrix(twave,net->Nifo);
    
    return(logL);
    
}

double fourier_nwip(double *a, double *b, double *Sn, int imin, int imax, int N)
{
    int i, j, k;
    double arg, product;
    double test;
    double ReA, ReB, ImA, ImB;
    
    arg = 0.0;
    for(i=imin; i<imax; i++)
    {
        j = i;
        k = N-i;
        ReA = a[j]; ImA = a[k];
        ReB = b[j]; ImB = b[k];
        product = ReA*ReB + ImA*ImB;
        arg += product/(Sn[i]);
    }
    
    return(4.0*arg);
    
}

double skylike(struct Net *net, Detector *network, double *sky, double *D, double *H, double **DHc,  double **DHs, double dt, int nt, int flag)
{
    int i, j, k, l, id, ifo;
    double alpha, delta, dphi, t0, A, FA, ecc, ciota, psi;
    double *dtimes, *F, *Fp, *Fc, *lambda;
    double tdelay, tx, toff, dc, ds, DH;
    double dcc, dss;
    double Ap, Ac;
    double clam, slam, x;
    double cphi, sphi;
    double logL;
    
    logL = 0.0;
    
        dtimes = (double*)malloc(sizeof(double)*DETECTOR_CATALOG_SIZE);
        F = (double*)malloc(sizeof(double)*net->Nifo);
        Fc = (double*)malloc(sizeof(double)*DETECTOR_CATALOG_SIZE);
        Fp = (double*)malloc(sizeof(double)*DETECTOR_CATALOG_SIZE);
        lambda = (double*)malloc(sizeof(double)*net->Nifo);
        
        k = (nt-1)/2;
        
        alpha = sky[0];
        delta = sky[1];
        psi = sky[2];
        ciota = sky[3];
        
        Ap = (1.0+ciota*ciota)/2.0;
        Ac = -ciota;
        
        A = sky[4];
        dphi = sky[5];
        toff = sky[6];
        
        cphi = cos(dphi);
        sphi = sin(dphi);
    
        Response(network, alpha, delta, psi, net->GMST, Fp, Fc, dtimes);
        
        for(id = 0; id<net->Nifo; id++)
        {
            ifo = net->labels[id];
            F[id] = sqrt(Ap*Ap*Fp[ifo]*Fp[ifo]+Ac*Ac*Fc[ifo]*Fc[ifo]);  // magnitude of response
            lambda[id] = atan2(Ac*Fc[ifo],Ap*Fp[ifo]);
            if(lambda[id] < 0.0) lambda[id] += 2.0*M_PI;
        }
        
        logL = 0.0;
        
        for(id = 0; id<net->Nifo; id++)
        {
            // everything is reference to geocenter
            ifo = net->labels[id];
            tdelay = toff+dtimes[ifo];
            
            i = (int)(floor(tdelay/dt));
            
            tx = (tdelay/dt - (double)(i));
            
            // if (flag == 1) printf("%d %d %d %f %f\n", i, i+k, nt, tx, tdelay/dt);
            
            // linear interpolation
            i += k;
            
            slam = sin(lambda[id]);
            clam = cos(lambda[id]);
            
            FA = A*F[id];
            
            if(i >= 0 && i < nt-2)
            {
                dc = DHc[id][i]*(1.0-tx)+DHc[id][i+1]*tx;
                ds = DHs[id][i]*(1.0-tx)+DHs[id][i+1]*tx;
                
                
                // put in phase rotation
                dcc = cphi*dc+sphi*ds;
                dss = -sphi*dc+cphi*ds;
                
                DH = clam*dcc+slam*dss;
                
                // relative likelihood
                x = -(FA*FA*H[id]-2.0*FA*DH)/2.0;
                
                // printf("%d %d %d %f %f %f %f %f %f\n", id, i, nt, tx, tdelay, x, FA, H[id], DH);
                
                logL += x;
                
            }
        }
        
    free(dtimes);
    free(F);
    free(Fc);
    free(Fp);
    free(lambda);
    
    return(logL);
    
}

double f_nwip(double *a, double *b, int n)
{
    int i, j, k;
    double arg, product;
    double test;
    double ReA, ReB, ImA, ImB;
    
    arg = 0.0;
    for(i=1; i<n/2; i++)
    {
        j = i;
        k = n-j;
        ReA = a[j]; ImA = a[k];
        ReB = b[j]; ImB = b[k];
        product = ReA*ReB + ImA*ImB;
        arg += product;
    }
    
    return(arg);
    
}


void skylikesetup(struct Net *net, double **data,  double **wave, double *D, double *H, double **DHc,  double **DHs, double Tobs, int n, int nt)
{
    double *corr, *corrf;
    double dt;
    int i, j, k, kk, l, ii, id;
    
    dt = Tobs/(double)(n);
    
    corr = double_vector(n);
    corrf = double_vector(n);
    
    for(id = 0; id<net->Nifo; id++)
    {
        
        D[id] = 4.0*f_nwip(data[id], data[id], n);
        H[id] = 4.0*f_nwip(wave[id], wave[id], n);
        
        corr[0] = 0.0;
        corr[n/2] = 0.0;
        corrf[0] = 0.0;
        corrf[n/2] = 0.0;
        for (i=1; i < n/2; i++)
        {
            l=i;
            k=n-i;
            corr[l]    = ( data[id][l]*wave[id][l] + data[id][k]*wave[id][k]);
            corr[k]    = ( data[id][k]*wave[id][l] - data[id][l]*wave[id][k]);
            corrf[l] = corr[k];
            corrf[k] = -corr[l];
        }
        
        
        gsl_fft_halfcomplex_radix2_inverse(corr, 1, n);
        gsl_fft_halfcomplex_radix2_inverse(corrf, 1, n);
        
        
        k = (nt-1)/2;
        
        for(i=-k; i<k; i++)
        {
            j = i+k;
            
            if(i < 0)
            {
                DHc[id][j] = 2.0*(double)(n)*corr[n+i];
                DHs[id][j] = 2.0*(double)(n)*corrf[n+i];
            }
            else
            {
                DHc[id][j] = 2.0*(double)(n)*corr[i];
                DHs[id][j] = 2.0*(double)(n)*corrf[i];
            }
            
        }
        
        //printf("%d %e %e %e %e\n", id, D[id], H[id], DHc[id][k], DHs[id][k]);
    
    }
    
    free_double_vector(corr);
    free_double_vector(corrf);
    
    
    
}

void fisherskysetup(struct Net *net, double **wave, double **HH, double Tobs, int n)
{
    double f;
    int i, j, k, l, bn, ii, id;
    double **wavef;
    
    wavef = double_matrix(net->Nifo,n);
    
    for(id = 0; id<net->Nifo; id++)
    {
        wavef[id][0] = 0.0;
        wavef[id][n/2] = 0.0;
        for(i = 1; i<n/2; i++)
        {
            f = (double)(i)/Tobs;
            j = i;
            k = n-i;
            wavef[id][j] = f*wave[id][j];
            wavef[id][k] = f*wave[id][k];
        }
    }
    
    for(id = 0; id<net->Nifo; id++)
    {
        HH[id][0] =  4.0*f_nwip(wave[id], wave[id], n);
        HH[id][1] =  4.0*f_nwip(wave[id], wavef[id], n);
        HH[id][2] =  4.0*f_nwip(wavef[id], wavef[id], n);
    }
    
    free_double_matrix(wavef,net->Nifo);
    
}

void fisher_skyproposal(gsl_rng * r, double **skyvecs, double *skyevals, double *jump)
{
    int a, j;
    double scale;
    
    a = (int)(gsl_rng_uniform(r)*(double)(NS));
    scale = gsl_ran_gaussian(r,1.0)*skyevals[a];
    
    for (j=0; j<NS; j++) jump[j] = scale*skyvecs[a][j];
}

void FisherEvec(double **fish, double *ej, double **ev, int d)
{
    int i, j, ec;
    
    ec = 0;
    for (i = 0 ; i < d ; i++) if(fabs(fish[i][i]) < 1.0e-16) ec = 1;
    
    if(ec == 0)
    {
        gsl_matrix *m = gsl_matrix_alloc(d, d);
        gsl_vector *eval = gsl_vector_alloc(d);
        gsl_matrix *evec = gsl_matrix_alloc(d, d);
        gsl_eigen_symmv_workspace *w = gsl_eigen_symmv_alloc(d);
        
        for (i = 0 ; i < d ; i++)
        {
            for (j = 0 ; j < d ; j++) gsl_matrix_set(m, i, j, fish[i][j]);
        }
        
        ec = gsl_eigen_symmv(m, eval, evec, w);
        gsl_eigen_symmv_free(w);
        gsl_eigen_symmv_sort(eval, evec, GSL_EIGEN_SORT_ABS_ASC);
        
        for (i = 0; i < d; i++)
        {
            ej[i] = gsl_vector_get(eval, i);
            for (j = 0 ; j < d ; j++) ev[i][j] = gsl_matrix_get(evec, j, i);
        }
        
        for (i = 0; i < d; i++)
        {
            if(ej[i] < 10.0) ej[i] = 10.0;
            ej[i] = 1.0/sqrt(ej[i]);
        }
        
        gsl_matrix_free(m);
        gsl_vector_free(eval);
        gsl_matrix_free(evec);
    }
    else
    {
        for (i = 0; i < d; i++)
        {
            ej[i] = 10000.0;
            for (j = 0 ; j < d ; j++)
            {
                ev[i][j] = 0.0;
                if(i==j) ev[i][j] = 1.0;
            }
        }
    }
}

void fisher_matrix_fastsky(struct Net *net, Detector *network, double *params, double **fisher, double **HH)
{
    int i, j, id, ifo;
    double alpha, delta, psi, ciota, A;
    double dtimesP[DETECTOR_CATALOG_SIZE], dtimesM[DETECTOR_CATALOG_SIZE];
    double dt0[DETECTOR_CATALOG_SIZE], dt1[DETECTOR_CATALOG_SIZE];
    double Fp0[DETECTOR_CATALOG_SIZE], Fc0[DETECTOR_CATALOG_SIZE], dtbase[DETECTOR_CATALOG_SIZE];
    double FpP[DETECTOR_CATALOG_SIZE], FcP[DETECTOR_CATALOG_SIZE], dtP[DETECTOR_CATALOG_SIZE];
    double FpM[DETECTOR_CATALOG_SIZE], FcM[DETECTOR_CATALOG_SIZE], dtM[DETECTOR_CATALOG_SIZE];
    double Fplus, Fcross, epss;
    double *dFplus, *dFcross;
    double *dlambda, *dtoff, *dF;
    double F, x, y, z;
    double Ap, Ac;
    
    dFplus = double_vector(NS);
    dFcross = double_vector(NS);
    dF = double_vector(NS);
    dlambda = double_vector(NS);
    dtoff = double_vector(NS);
    
    for(i = 0; i<NS; i++)
    {
        dFplus[i] = 0.0;
        dFcross[i] = 0.0;
        dF[i] = 0.0;
        dlambda[i] = 0.0;
        dtoff[i] = 0.0;
    }
    
    dlambda[5] = 1.0;
    dtoff[6] = -1.0;
    
    alpha = params[0];
    delta = params[1];
    psi = params[2];
    ciota = params[3];
    A = params[4];
    
    Ap = 0.5*(1.0+ciota*ciota);
    Ac = -ciota;
    epss = 1.0e-6;
    
    Response(network, alpha+epss, delta, psi, net->GMST, FpP, FcP, dtimesP);
    Response(network, alpha-epss, delta, psi, net->GMST, FpM, FcM, dtimesM);
    for(id = 0; id<net->Nifo; id++)
    {
        ifo = net->labels[id];
        dt0[id] = (dtimesP[ifo]-dtimesM[ifo])/(2.0*epss);
    }
    
    Response(network, alpha, delta+epss, psi, net->GMST, FpP, FcP, dtimesP);
    Response(network, alpha, delta-epss, psi, net->GMST, FpM, FcM, dtimesM);
    for(id = 0; id<net->Nifo; id++)
    {
        ifo = net->labels[id];
        dt1[id] = (dtimesP[ifo]-dtimesM[ifo])/(2.0*epss);
    }
    
    for(i = 0; i<NS; i++)
    {
        for(j = 0; j<NS; j++) fisher[i][j] = 0.0;
    }
    
    Response(network, alpha, delta, psi, net->GMST, Fp0, Fc0, dtbase);
    
    for(id = 0; id<net->Nifo; id++)
    {
        ifo = net->labels[id];
        dtoff[0] = -dt0[id];
        dtoff[1] = -dt1[id];
        
        Fplus = Fp0[ifo];
        Fcross = Fc0[ifo];
        F = A*sqrt(Ap*Ap*Fplus*Fplus+Ac*Ac*Fcross*Fcross);
        
        Response(network, alpha+epss, delta, psi, net->GMST, FpP, FcP, dtP);
        Response(network, alpha-epss, delta, psi, net->GMST, FpM, FcM, dtM);
        dFplus[0] = (FpP[ifo]-FpM[ifo])/(2.0*epss);
        dFcross[0] = (FcP[ifo]-FcM[ifo])/(2.0*epss);
        
        Response(network, alpha, delta+epss, psi, net->GMST, FpP, FcP, dtP);
        Response(network, alpha, delta-epss, psi, net->GMST, FpM, FcM, dtM);
        dFplus[1] = (FpP[ifo]-FpM[ifo])/(2.0*epss);
        dFcross[1] = (FcP[ifo]-FcM[ifo])/(2.0*epss);
        
        Response(network, alpha, delta, psi+epss, net->GMST, FpP, FcP, dtP);
        Response(network, alpha, delta, psi-epss, net->GMST, FpM, FcM, dtM);
        dFplus[2] = (FpP[ifo]-FpM[ifo])/(2.0*epss);
        dFcross[2] = (FcP[ifo]-FcM[ifo])/(2.0*epss);
        
        x = A*A/F;
        dF[0] = x*(Ap*Ap*Fplus*dFplus[0]+Ac*Ac*Fcross*dFcross[0]);
        dF[1] = x*(Ap*Ap*Fplus*dFplus[1]+Ac*Ac*Fcross*dFcross[1]);
        dF[2] = x*(Ap*Ap*Fplus*dFplus[2]+Ac*Ac*Fcross*dFcross[2]);
        dF[3] = x*(Ap*ciota*Fplus*Fplus+Ac*Fcross*Fcross);
        dF[4] = F/A;
        
        x = A*A/(F*F);
        dlambda[0] = x*Ap*Ac*(Fplus*dFcross[0]-Fcross*dFplus[0]);
        dlambda[1] = x*Ap*Ac*(Fplus*dFcross[1]-Fcross*dFplus[1]);
        dlambda[2] = x*Ap*Ac*(Fplus*dFcross[2]-Fcross*dFplus[2]);
        dlambda[3] = x*Fcross*Fplus*(Ap-Ac*ciota);
        
        x = F*F;
        y = TPI*x;
        z = TPI*y;
        for(i = 0; i<NS; i++)
        {
            for(j = 0; j<NS; j++)
            {
                fisher[i][j] += HH[id][0]*(dF[i]*dF[j]+x*dlambda[i]*dlambda[j])
                              + HH[id][1]*y*(dlambda[i]*dtoff[j]+dlambda[j]*dtoff[i])
                              + HH[id][2]*z*dtoff[i]*dtoff[j];
            }
        }
    }
    
    free_double_vector(dFplus);
    free_double_vector(dFcross);
    free_double_vector(dF);
    free_double_vector(dlambda);
    free_double_vector(dtoff);
}

void Ring(Detector *network, double *skyx, double *skyy, int d1, int d2, double GMST, gsl_rng * r)
{
    int i;
    double zmag, cmu, smu, rot, cr, sr, gha, cosd;
    Vector3D b, z, w, u, v, wr;
    
    b.x = network[d1].vertex.x - network[d2].vertex.x;
    b.y = network[d1].vertex.y - network[d2].vertex.y;
    b.z = network[d1].vertex.z - network[d2].vertex.z;
    zmag = sqrt(b.x*b.x + b.y*b.y + b.z*b.z);
    z.x = b.x/zmag;
    z.y = b.y/zmag;
    z.z = b.z/zmag;
    
    gha = GMST - skyx[0];
    w.x = -cos(skyx[1]) * cos(gha);
    w.y =  cos(skyx[1]) * sin(gha);
    w.z = -sin(skyx[1]);
    
    cmu = z.x*w.x + z.y*w.y + z.z*w.z;
    if(cmu > 0.999999) cmu = 0.999999;
    if(cmu < -0.999999) cmu = -0.999999;
    smu = sqrt(1.0-cmu*cmu);
    
    u.x = (w.x - cmu*z.x)/smu;
    u.y = (w.y - cmu*z.y)/smu;
    u.z = (w.z - cmu*z.z)/smu;
    
    v.x = (z.y*w.z - z.z*w.y)/smu;
    v.y = (z.z*w.x - z.x*w.z)/smu;
    v.z = (z.x*w.y - z.y*w.x)/smu;
    
    rot = TPI*gsl_rng_uniform(r);
    cr = cos(rot);
    sr = sin(rot);
    
    wr.x = cmu*z.x + smu*(cr*u.x + sr*v.x);
    wr.y = cmu*z.y + smu*(cr*u.y + sr*v.y);
    wr.z = cmu*z.z + smu*(cr*u.z + sr*v.z);
    
    skyy[1] = asin(-wr.z);
    cosd = cos(skyy[1]);
    if(fabs(cosd) < 1.0e-12) cosd = (cosd < 0.0) ? -1.0e-12 : 1.0e-12;
    gha = atan2(wr.y/cosd, -wr.x/cosd);
    skyy[0] = GMST - gha;
    while(skyy[0] < 0.0) skyy[0] += TPI;
    while(skyy[0] > TPI) skyy[0] -= TPI;
    
    for(i = 2; i < NS; i++) skyy[i] = skyx[i];
}

void DetectorPlaneAntipode(Detector *network, double *skyx, double *skyy, int d1, int d2, int d3, double GMST)
{
    int i;
    double gha, ndot, nmag, cosd, zarg;
    Vector3D b12, b13, n, w, wr;
    
    for(i = 0; i < NS; i++) skyy[i] = skyx[i];
    
    b12.x = network[d2].vertex.x - network[d1].vertex.x;
    b12.y = network[d2].vertex.y - network[d1].vertex.y;
    b12.z = network[d2].vertex.z - network[d1].vertex.z;
    
    b13.x = network[d3].vertex.x - network[d1].vertex.x;
    b13.y = network[d3].vertex.y - network[d1].vertex.y;
    b13.z = network[d3].vertex.z - network[d1].vertex.z;
    
    n.x = b12.y*b13.z - b12.z*b13.y;
    n.y = b12.z*b13.x - b12.x*b13.z;
    n.z = b12.x*b13.y - b12.y*b13.x;
    nmag = sqrt(n.x*n.x + n.y*n.y + n.z*n.z);
    
    if(nmag <= 0.0) return;
    
    n.x /= nmag;
    n.y /= nmag;
    n.z /= nmag;
    
    gha = GMST - skyx[0];
    w.x = -cos(skyx[1]) * cos(gha);
    w.y =  cos(skyx[1]) * sin(gha);
    w.z = -sin(skyx[1]);
    
    // Reflect the propagation vector through the detector plane.
    ndot = w.x*n.x + w.y*n.y + w.z*n.z;
    wr.x = w.x - 2.0*ndot*n.x;
    wr.y = w.y - 2.0*ndot*n.y;
    wr.z = w.z - 2.0*ndot*n.z;
    wr = normalize(wr);
    
    zarg = -wr.z;
    if(zarg > 1.0) zarg = 1.0;
    if(zarg < -1.0) zarg = -1.0;
    
    skyy[1] = asin(zarg);
    cosd = cos(skyy[1]);
    if(fabs(cosd) > 1.0e-12)
    {
        gha = atan2(wr.y/cosd, -wr.x/cosd);
        skyy[0] = GMST - gha;
        while(skyy[0] < 0.0) skyy[0] += TPI;
        while(skyy[0] > TPI) skyy[0] -= TPI;
    }
}

void uvwz(double *u, double *v, double *w, double *z, double *params)
{
    double psi, ciota, ecc, Amp, phi;
    double cphi, sphi, c2p, s2p;
    double Ap, Ac;
    
    psi = params[2];
    ciota = params[3];
    Amp = params[4];
    phi  = params[5];
    
    Ap = 0.5*(1.0+ciota*ciota);
    Ac = -ciota;
    ecc = Ac/Ap;
    Amp *= Ap;
    
    cphi = cos(phi);
    sphi = sin(phi);
    c2p = cos(2.0*psi);
    s2p = sin(2.0*psi);
    
    *u = Amp*(cphi*c2p+ecc*sphi*s2p);
    *v = Amp*(cphi*s2p-ecc*sphi*c2p);
    *w = Amp*(sphi*c2p-ecc*cphi*s2p);
    *z = Amp*(sphi*s2p+ecc*cphi*c2p);
}

void uvwz_sol(double *uy, double *vy, double *wy, double *zy, double ux, double vx, double wx, double zx,
              double fp1x, double fp1y, double fc1x, double fc1y, double fp2x, double fp2y, double fc2x, double fc2y)
{
    double den = fc1y*fp2y-fc2y*fp1y;
    
    *uy = (vx*(fc1y*fc2x-fc1x*fc2y)+ux*(fc1y*fp2x - fc2y*fp1x))/den;
    *vy = (vx*(fc1x*fp2y-fc2x*fp1y)+ux*(fp2y*fp1x-fp1y*fp2x))/den;
    *wy = (zx*(fc1y*fc2x-fc1x*fc2y)+wx*(fc1y*fp2x - fc2y*fp1x))/den;
    *zy = (zx*(fc1x*fp2y-fc2x*fp1y)+wx*(fp2y*fp1x-fp1y*fp2x))/den;
}

void exsolve(double *phiy, double *psiy, double *Ay, double *ciotay, double uy, double vy, double wy, double zy)
{
    double q, rad, x, p, ecc, ci;
    int flag1, flag2, flag3;
    
    q = 2.0*(uy*wy+vy*zy)/((uy*uy+vy*vy) - (wy*wy+zy*zy));
    rad = sqrt(1.0+q*q);
    
    x = atan2(2.0*(uy*wy+vy*zy), ((uy*uy+vy*vy) - (wy*wy+zy*zy)));
    if(x < 0.0) x += TPI;
    *phiy = 0.5*x;
    
    flag1 = (cos(x) > 0.0);
    flag2 = (cos(x/2.0) > 0.0);
    flag3 = (sin(x/2.0) > 0.0);
    
    p = ((rad+1.0)*(uy*uy+vy*vy)+(rad-1.0)*(wy*wy+zy*zy)+2.0*q*(uy*wy+vy*zy))/(2.0*rad);
    
    if (flag1 == 1) ecc = (uy*zy-vy*wy)/p;
    else ecc = p/(uy*zy-vy*wy);
    
    ci = -(1.0-sqrt(1.0-ecc*ecc))/ecc;
    *ciotay = ci;
    
    if(flag1 == 1 && flag2 == 1) x = atan2((1.0+rad)*vy+q*zy,(1.0+rad)*uy+q*wy);
    else if (flag1 == 1 && flag2 == 0) x = atan2(-((1.0+rad)*vy+q*zy),-((1.0+rad)*uy+q*wy));
    else if (flag1 == 0 && flag3==1) x = atan2(ecc*((1.0+rad)*uy+q*wy), -ecc*((1.0+rad)*vy+q*zy));
    else x = atan2(-ecc*((1.0+rad)*uy+q*wy), ecc*((1.0+rad)*vy+q*zy));
    
    if(x < 0.0) x += TPI;
    *psiy = 0.5*x;
    
    if(flag1 == 1) x = sqrt(p);
    else x = sqrt(p)/fabs(ecc);
    
    *Ay = 2.0*x/(1.0+ci*ci);
}

int skymap(Detector *network, double *paramsx, double *paramsy, double GMST, int ifo1, int ifo2, int iref)
{
    double fxplus[DETECTOR_CATALOG_SIZE], fxcross[DETECTOR_CATALOG_SIZE], dtimesx[DETECTOR_CATALOG_SIZE];
    double fyplus[DETECTOR_CATALOG_SIZE], fycross[DETECTOR_CATALOG_SIZE], dtimesy[DETECTOR_CATALOG_SIZE];
    double ux, vx, wx, zx;
    double uy, vy, wy, zy;
    double phiy, psiy, Ay, ciotay;
    double den, row1, row2;
    const double singular_tol = 1.0e-3;
    
    Response(network, paramsx[0], paramsx[1], 0.0, GMST, fxplus, fxcross, dtimesx);
    Response(network, paramsy[0], paramsy[1], 0.0, GMST, fyplus, fycross, dtimesy);
    
    den = fycross[ifo1]*fyplus[ifo2]-fycross[ifo2]*fyplus[ifo1];
    row1 = sqrt(fyplus[ifo1]*fyplus[ifo1]+fycross[ifo1]*fycross[ifo1]);
    row2 = sqrt(fyplus[ifo2]*fyplus[ifo2]+fycross[ifo2]*fycross[ifo2]);
    
    if(row1 <= 0.0 || row2 <= 0.0 || fabs(den) < singular_tol*row1*row2) return 0;
    
    uvwz(&ux, &vx, &wx, &zx, paramsx);
    uvwz_sol(&uy, &vy, &wy, &zy, ux, vx, wx, zx,
             fxplus[ifo1], fyplus[ifo1], fxcross[ifo1], fycross[ifo1],
             fxplus[ifo2], fyplus[ifo2], fxcross[ifo2], fycross[ifo2]);
    exsolve(&phiy, &psiy, &Ay, &ciotay, uy, vy, wy, zy);
    
    if(!isfinite(phiy) || !isfinite(psiy) || !isfinite(Ay) || !isfinite(ciotay)) return 0;
    if(Ay <= 0.0 || fabs(ciotay) > 1.0) return 0;
    
    paramsy[2] = psiy;
    paramsy[3] = ciotay;
    paramsy[4] = Ay;
    paramsy[5] = phiy;
    paramsy[6] = paramsx[6]+dtimesx[ifo1]-dtimesy[ifo1];
    
    (void)iref;
    
    return 1;
}

double skydensity(Detector *network, double *paramsx, double *paramsy, double GMST, int ifo1, int ifo2, int iref)
{
    int i, j;
    double *paramsxp, *paramsyp;
    double *Jmat;
    double Jack, ep;
    
    ep = 1.0e-6;
    paramsxp = double_vector(NS);
    paramsyp = double_vector(NS);
    Jmat = double_vector(16);
    
    if(!skymap(network, paramsx, paramsy, GMST, ifo1, ifo2, iref))
    {
        free_double_vector(paramsyp);
        free_double_vector(paramsxp);
        free_double_vector(Jmat);
        return 0.0;
    }
    
    for(i=0; i< 4; i++)
    {
        for(j=0; j<= 5; j++) paramsxp[j] = paramsx[j];
        for(j=0; j<= 1; j++) paramsyp[j] = paramsy[j];
        paramsxp[i+2] += ep;
        
        if(!skymap(network, paramsxp, paramsyp, GMST, ifo1, ifo2, iref))
        {
            free_double_vector(paramsyp);
            free_double_vector(paramsxp);
            free_double_vector(Jmat);
            return 0.0;
        }
        
        if( fabs(paramsyp[2]+M_PI/2.0-paramsy[2]) < fabs(paramsyp[2]-paramsy[2]) ) paramsyp[2] += M_PI/2.0;
        if( fabs(paramsyp[2]-M_PI/2.0-paramsy[2]) < fabs(paramsyp[2]-paramsy[2]) ) paramsyp[2] -= M_PI/2.0;
        if( fabs(paramsyp[5]+M_PI-paramsy[5]) < fabs(paramsyp[5]-paramsy[5]) ) paramsyp[5] += M_PI;
        if( fabs(paramsyp[5]-M_PI-paramsy[5]) < fabs(paramsyp[5]-paramsy[5]) ) paramsyp[5] -= M_PI;
        
        for(j=0; j< 4; j++) Jmat[i*4+j] = (paramsyp[j+2]-paramsy[j+2])/ep;
    }
    
    Jack = fabs(det(Jmat, 4));
    if(!isfinite(Jack)) Jack = 0.0;
    
    free_double_vector(paramsyp);
    free_double_vector(paramsxp);
    free_double_vector(Jmat);
    
    return Jack;
}

void dshifts(struct Net *net, Detector *network, double *sky, double *params)
{
    int id, ifo;
    double dtimes[DETECTOR_CATALOG_SIZE], Fplus[DETECTOR_CATALOG_SIZE], Fcross[DETECTOR_CATALOG_SIZE];
    double F[DETECTOR_CATALOG_SIZE], lambda[DETECTOR_CATALOG_SIZE];
    double ecc;
    
    Response(network, sky[0], sky[1], sky[2], net->GMST, Fplus, Fcross, dtimes);
    
    for(id = 0; id<net->Nifo; id++)
    {
        ifo = net->labels[id];
        ecc = -2.0*sky[3]/(1.0+sky[3]*sky[3]);
        F[id] = sqrt(Fplus[ifo]*Fplus[ifo]+ecc*ecc*Fcross[ifo]*Fcross[ifo]);
        lambda[id] = atan2(ecc*Fcross[ifo],Fplus[ifo]);
        if(lambda[id] < 0.0) lambda[id] += TPI;
    }
    
    for(id = 1; id<net->Nifo; id++)
    {
        params[(id-1)*3+NX] = lambda[id]-lambda[0];
        if(params[(id-1)*3+NX] < 0.0) params[(id-1)*3+NX] += TPI;
        if(params[(id-1)*3+NX] > TPI) params[(id-1)*3+NX] -= TPI;
        params[(id-1)*3+NX+1] = dtimes[net->labels[id]]-dtimes[net->labels[0]];
        params[(id-1)*3+NX+2] = F[id]/F[0];
    }
}

double det(double *A, int N)
{
    int i, j, s;
    double dx;
    gsl_matrix *m = gsl_matrix_alloc(N, N);
    gsl_permutation *p = gsl_permutation_alloc(N);
    
    for(i = 0; i < N; i++)
    {
        for(j = 0; j < N; j++) gsl_matrix_set(m, i, j, A[j*N+i]);
    }
    
    gsl_linalg_LU_decomp(m, p, &s);
    
    dx = 1.0;
    for(i = 0; i < N; i++) dx *= gsl_matrix_get(m, i, i);
    dx = fabs(dx);
    
    gsl_permutation_free(p);
    gsl_matrix_free(m);
    
    return dx;
}






void fulltemplates(struct Net *net, Detector *network, double **hwave, RealVector *freq, double *params, int N)
{
    
    AmpPhaseFDWaveform *ap = NULL;
    double phi0, fRef_in, mc, q, m1_SI, m2_SI, chi1, chi2, f_min, f_max, distance;
    int ret, flag, i, j, id, ifo;
    double p, cp, sp;
    double f, x, y, deltaF, ts, Tobs, sqT;
    double pd, Ar, td, A;
    double alpha, delta, psi, ciota, lambda;
    double Ap, Ac;
    double Fplus, Fcross, Fs;
    double mt, eta, dm, fs;
    double *dtimes, *Fp, *Fc;
    
    dtimes = (double*)malloc(sizeof(double)*DETECTOR_CATALOG_SIZE);
    Fp = (double*)malloc(sizeof(double)*DETECTOR_CATALOG_SIZE);
    Fc = (double*)malloc(sizeof(double)*DETECTOR_CATALOG_SIZE);
    
    fs = fbegin(params);
    //printf("start frequency %f\n", fs);
    
    Tobs = 1.0/freq->data[1];
    sqT = sqrt(Tobs);
    
    m1_SI = params[0]*MSUN_SI;
    m2_SI = params[1]*MSUN_SI;

    chi1 = params[2];
    chi2 = params[3];
    
    phi0 = 0.5*params[4];  // [4] is the GW phase, while PhenomD uses orbital phase
    ts = Tobs-params[5];
    
    distance = exp(params[6]);
    
    fRef_in = fref;
    
    ret = IMRPhenomDGenerateh22FDAmpPhase(&ap, freq, 0, phi0, fRef_in, m1_SI, m2_SI, chi1, chi2, distance);
    
    alpha = params[7];
    delta = params[8];
    psi = params[9];
    ciota = params[10];
    
    Ap = (1.0+ciota*ciota)/2.0;
    Ac = -ciota;
 
    Response(network, alpha, delta, psi, net->GMST, Fp, Fc, dtimes);
    
    for (id=0; id< net->Nifo; id++)
    {
        ifo = net->labels[id];
        //printf("%d %f %f %e\n", id, Fp[ifo], Fc[ifo], dtimes[ifo]);
        Fs = sqrt(Ap*Ap*Fp[ifo]*Fp[ifo]+Ac*Ac*Fc[ifo]*Fc[ifo]);  // magnitude of response
        lambda = atan2(Ac*Fc[ifo],Ap*Fp[ifo]);
        if(lambda < 0.0) lambda += 2.0*M_PI;
        td = dtimes[ifo];
        
        hwave[id][0] = 0.0;
        hwave[id][N/2] = 0.0;
        
        for (i=1; i< N/2; i++)
        {
            f = freq->data[i];
            if(f > fs)
            {
            A = Fs*h22fac*ap->amp[i]/sqT;
            p = ap->phase[i];
            x = 2.0*M_PI*f*(ts-td)+lambda-p;
            hwave[id][i] = A*cos(x);
            hwave[id][N-i] = A*sin(x);
            }
            else
            {
                hwave[id][i] = 0.0;
                hwave[id][N-i] = 0.0;
            }
        }
    }
    
    free(dtimes);
    free(Fp);
    free(Fc);
    DestroyAmpPhaseFDWaveform(ap);
    
}


void setup_network(Detector *network)
{
    // 1. LIGO-Hanford (H1)
    network[0].name = "LIGO-Hanford (H1)";
    network[0].vertex = (Vector3D){ -2161414.92636, -3834695.17889, 4600350.22664 };
    network[0].arm_X  = normalize((Vector3D){ -0.22389266154,  0.79983062746,  0.55690487831 });
    network[0].arm_Y  = normalize((Vector3D){ -0.91397818574,  0.02609403989, -0.40492342125 });
    
    // 2. LIGO-Livingston (L1)
    network[1].name = "LIGO-Livingston (L1)";
    network[1].vertex = (Vector3D){ -74276.0447238, -5496283.71971, 3224257.01744 };
    network[1].arm_X  = normalize((Vector3D){ -0.95457412153, -0.14158077340, -0.26218911324 });
    network[1].arm_Y  = normalize((Vector3D){  0.29774156894, -0.48791033647, -0.82054461286 });
    
    // 3. LIGO-India (I1), nominal LAL/PyCBC LIO_4k geometry
    network[2].name = "LIGO-India (I1)";
    network[2].vertex = (Vector3D){ 1348971.15479, 5857428.26577, 2127569.25209 };
    network[2].arm_X  = normalize((Vector3D){  0.38496278183, -0.39387275094,  0.83466634811 });
    network[2].arm_Y  = normalize((Vector3D){  0.89838844906, -0.04722636126, -0.43665531647 });
    
    // 4. Virgo (V1), LAL 7.7.1 VIRGO geometry
    network[3].name = "Virgo (V1)";
    network[3].vertex = (Vector3D){ 4546374.09900, 842989.697626, 4378576.96241 };
    network[3].arm_X  = normalize((Vector3D){ -0.70045821457,  0.20848949018,  0.68256166178 });
    network[3].arm_Y  = normalize((Vector3D){ -0.05379254404, -0.96908180835,  0.24080450769 });
}

void Response(Detector *network, double ra_rad, double dec_rad, double psi_rad, double gmst, double *Fp, double *Fc, double *delay)
{
    double gha = gmst - ra_rad;
    
    // Unit propagation vector pointing FROM the source TO Earth
    Vector3D w;
    w.x = -cos(dec_rad) * cos(gha);
    w.y =  cos(dec_rad) * sin(gha);
    w.z = -sin(dec_rad);
    
    // Unrotated baseline Wave Frame axes (Meridian and Parallel lines)
    Vector3D p_init = { -sin(dec_rad) * cos(gha),  sin(dec_rad) * sin(gha), cos(dec_rad) };
    Vector3D q_init = {  sin(gha),                 cos(gha),                0.0          };
    
    // Polarization axes matching the PyCBC/LAL x-y wave-frame convention.
    Vector3D u_x, u_y;
    u_x.x = -cos(psi_rad) * q_init.x + sin(psi_rad) * p_init.x;
    u_x.y = -cos(psi_rad) * q_init.y + sin(psi_rad) * p_init.y;
    u_x.z = -cos(psi_rad) * q_init.z + sin(psi_rad) * p_init.z;
    
    u_y.x =  sin(psi_rad) * q_init.x + cos(psi_rad) * p_init.x;
    u_y.y =  sin(psi_rad) * q_init.y + cos(psi_rad) * p_init.y;
    u_y.z =  sin(psi_rad) * q_init.z + cos(psi_rad) * p_init.z;
    
    for(int i = 0; i < DETECTOR_CATALOG_SIZE; i++)
    {
        Fp[i]  = compute_response(network[i].arm_X, network[i].arm_Y, u_x, u_y);
        Fc[i] = compute_cross_response(network[i].arm_X, network[i].arm_Y, u_x, u_y);
        delay[i]   = compute_geocenter_delay(network[i].vertex, w);
    }

}


void geotemplate(double *gwave, RealVector *freq, double *params, int N)
{
    
    AmpPhaseFDWaveform *ap = NULL;
    double phi0, fRef_in, mc, q, m1_SI, m2_SI, chi1, chi2, f_min, f_max, distance;
    int ret, flag, i, j, id;
    double p, cp, sp;
    double f, x, y, deltaF, ts, Tobs, sqT;
    double pd, Ar, td, A;
    double mt, eta, dm, fs;
    
    fs = fbegin(params);
    
    Tobs = 1.0/freq->data[1];
    sqT = sqrt(Tobs);
    
    m1_SI = params[0]*MSUN_SI;
    m2_SI = params[1]*MSUN_SI;
    
    chi1 = params[2];
    chi2 = params[3];
    
    phi0 = 0.5*params[4];  // I'm holding the GW phase in [4], while PhenomD wants orbital
    ts = Tobs-params[5];
    
    distance = exp(params[6]);
    
    fRef_in = fref;
    
    ret = IMRPhenomDGenerateh22FDAmpPhase(&ap, freq, 0, phi0, fRef_in, m1_SI, m2_SI, chi1, chi2, distance);

        gwave[0] = 0.0;
        gwave[N/2] = 0.0;
        
        for (i=1; i< N/2; i++)
        {
            f = freq->data[i];
            if(f > fs)
            {
            A = h22fac*ap->amp[i]/sqT;
            p = ap->phase[i];
            x = 2.0*M_PI*f*ts-p;
            gwave[i] = A*cos(x);
            gwave[N-i] = A*sin(x);
            }
            else
            {
                gwave[i] = 0.0;
                gwave[N-i] = 0.0;
            }
        }
    
    DestroyAmpPhaseFDWaveform(ap);
    
}

// Normalizes a 3D vector to unit length
Vector3D normalize(Vector3D v) {
    double len = sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    Vector3D u = {v.x / len, v.y / len, v.z / len};
    return u;
}

// Computes the dot product of two vectors
double dot_product(Vector3D a, Vector3D b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Computes the plus-polarization response tensor projection
double compute_response(Vector3D X, Vector3D Y, Vector3D P, Vector3D Q) {
    double xp = dot_product(X, P);
    double xq = dot_product(X, Q);
    double yp = dot_product(Y, P);
    double yq = dot_product(Y, Q);
    
    return 0.5 * ((xp * xp - xq * xq) - (yp * yp - yq * yq));
}

// Computes the cross-polarization response tensor projection
double compute_cross_response(Vector3D X, Vector3D Y, Vector3D P, Vector3D Q) {
    double xp = dot_product(X, P);
    double xq = dot_product(X, Q);
    double yp = dot_product(Y, P);
    double yq = dot_product(Y, Q);
    
    return xp * xq - yp * yq;
}

// Subroutine: Computes time delay between Earth Geocenter and Detector vertex (in seconds)
double compute_geocenter_delay(Vector3D vertex, Vector3D wave_prop) {
    // Delay = (Vertex . Wave_Propagation_Vector) / c
    return dot_product(vertex, wave_prop) / CLIGHT;
}


static int postprocess_sky_chain(const char *chain_file, long Nside, SkyPostSummary *summary)
{
    long i, k, Npix, Nsample, k90, boundary_count;
    double theta, phi, costh, x, y, level90;
    double logL, DL;
    double *map, *boundary, *region;
    char line[4096];
    FILE *in, *out;
    gsl_vector *mv;
    gsl_permutation *perm;
    const double pi = 0.5*TPI;
    
    if(summary != NULL)
    {
        summary->area50 = 0.0;
        summary->area90 = 0.0;
        summary->boundary_area = 0.0;
        summary->boundary_pixels = 0;
        summary->map_theta = 0.0;
        summary->map_phi = 0.0;
    }
    
    in = fopen(chain_file, "r");
    if(in == NULL)
    {
        fprintf(stderr, "Could not open chain file %s\n", chain_file);
        return 1;
    }
    
    Nsample = 0;
    while(fgets(line, sizeof(line), in) != NULL)
    {
        if(sscanf(line, "%ld%lf%lf%lf%lf", &k, &logL, &phi, &costh, &DL) == 5) Nsample++;
    }
    rewind(in);
    
    if(Nsample <= 0)
    {
        fprintf(stderr, "No sky samples found in %s\n", chain_file);
        fclose(in);
        return 1;
    }
    
    if(!skyhist_is_power_of_two(Nside))
    {
        fprintf(stderr, "Nside must be a power of 2 for nested-neighbor boundary finding\n");
        fclose(in);
        return 1;
    }
    
    Npix = 12*Nside*Nside;
    
    map = skyhist_double_vector(Npix);
    boundary = skyhist_double_vector(Npix);
    region = skyhist_double_vector(Npix);
    if(map == NULL || boundary == NULL || region == NULL)
    {
        fprintf(stderr, "Could not allocate sky maps for Nside=%ld\n", Nside);
        fclose(in);
        free(map);
        free(boundary);
        free(region);
        return 1;
    }
    
    for(i = 0; i < Npix; i++)
    {
        map[i] = 0.0;
        boundary[i] = 0.0;
        region[i] = 0.0;
    }
    
    out = fopen("skycheck.dat", "w");
    if(out == NULL)
    {
        fprintf(stderr, "Could not open skycheck.dat for writing\n");
        fclose(in);
        free(map);
        free(boundary);
        free(region);
        return 1;
    }
    
    i = 0;
    while(fgets(line, sizeof(line), in) != NULL && i < Nsample)
    {
        if(sscanf(line, "%ld%lf%lf%lf%lf", &k, &logL, &phi, &costh, &DL) != 5) continue;
        theta = acos(skyhist_clamp_double(costh, -1.0, 1.0));
        if(phi < 0.0) phi += TPI;
        if(phi > TPI) phi -= TPI;
        fprintf(out, "%ld %f %f\n", i, phi, theta);
        ang2pix_ring(Nside, theta, phi, &k);
        map[k] += 1.0;
        i++;
    }
    fclose(in);
    fclose(out);
    
    for(i = 0; i < Npix; i++) map[i] /= (double)(Nsample);
    
    mv = gsl_vector_alloc(Npix);
    perm = gsl_permutation_alloc(Npix);
    if(mv == NULL || perm == NULL)
    {
        fprintf(stderr, "Could not allocate sky-map sorting workspace\n");
        if(mv != NULL) gsl_vector_free(mv);
        if(perm != NULL) gsl_permutation_free(perm);
        free(map);
        free(boundary);
        free(region);
        return 1;
    }
    
    for(i = 0; i < Npix; i++) gsl_vector_set(mv, i, map[i]);
    gsl_sort_vector_index(perm, mv);
    
    k = 0;
    x = 0.0;
    while(x < 0.5 && k < Npix)
    {
        k++;
        x += gsl_vector_get(mv, perm->data[Npix-k]);
    }
    
    y = (double)(k)/(double)(Npix)*4.0*pi*(180.0/pi)*(180.0/pi);
    if(summary != NULL) summary->area50 = y;
    printf("50 percent credible interval = %f  square degrees\n", y);
    
    while(x < 0.9 && k < Npix)
    {
        k++;
        x += gsl_vector_get(mv, perm->data[Npix-k]);
    }
    
    k90 = k;
    if(k90 < Npix) level90 = gsl_vector_get(mv, perm->data[Npix-k90-1]);
    else level90 = 0.0;
    
    y = (double)(k)/(double)(Npix)*4.0*pi*(180.0/pi)*(180.0/pi);
    if(summary != NULL) summary->area90 = y;
    printf("90 percent credible interval = %f  square degrees\n", y);
    
    boundary_count = skyhist_make_credible_boundary(Nside, Npix, map, perm, k90, level90, boundary, region);
    if(boundary_count < 0)
    {
        gsl_permutation_free(perm);
        gsl_vector_free(mv);
        free(map);
        free(boundary);
        free(region);
        return 1;
    }
    y = (double)(boundary_count)/(double)(Npix)*4.0*pi*(180.0/pi)*(180.0/pi);
    if(summary != NULL)
    {
        summary->boundary_pixels = boundary_count;
        summary->boundary_area = y;
    }
    printf("90 percent credible boundary = %ld pixels = %f square degrees\n", boundary_count, y);
    
    pix2ang_ring(Nside, perm->data[Npix-1], &theta, &phi);
    if(summary != NULL)
    {
        summary->map_theta = theta*180.0/pi;
        summary->map_phi = phi*180.0/pi;
    }
    printf("MAP location theta = %f phi = %f\n", theta*180.0/pi, phi*180.0/pi);
    
    gsl_permutation_free(perm);
    gsl_vector_free(mv);
    
    if(skyhist_write_map("sky.dat", map, Npix) != 0 ||
       skyhist_write_map("sky90_region.dat", region, Npix) != 0 ||
       skyhist_write_map("sky90_boundary.dat", boundary, Npix) != 0)
    {
        free(map);
        free(boundary);
        free(region);
        return 1;
    }
    
    free(map);
    free(boundary);
    free(region);
    
    return 0;
}

static double *skyhist_double_vector(long N)
{
    return (double *)malloc((N+1)*sizeof(double));
}

static int *skyhist_int_vector(long N)
{
    return (int *)malloc((N+1)*sizeof(int));
}

static int skyhist_is_power_of_two(long n)
{
    return (n > 0 && (n & (n-1)) == 0);
}

static double skyhist_clamp_double(double x, double xmin, double xmax)
{
    if(x < xmin) return xmin;
    if(x > xmax) return xmax;
    return x;
}

static int skyhist_write_map(const char *filename, double *map, long Npix)
{
    long i;
    FILE *out;
    
    out = fopen(filename, "w");
    if(out == NULL)
    {
        fprintf(stderr, "Could not open %s for writing\n", filename);
        return 1;
    }
    
    for(i = 0; i < Npix; i++) fprintf(out, "%f\n", map[i]);
    fclose(out);
    
    return 0;
}

static long skyhist_make_credible_boundary(long Nside, long Npix, double *map, gsl_permutation *perm, long k90, double level90, double *boundary, double *region)
{
    long i, j, iring, inest, nneighbor, count;
    long neighbors[8];
    double *map_nest;
    double *boundary_nest;
    int *inside90;
    
    map_nest = skyhist_double_vector(Npix);
    boundary_nest = skyhist_double_vector(Npix);
    inside90 = skyhist_int_vector(Npix);
    if(map_nest == NULL || boundary_nest == NULL || inside90 == NULL)
    {
        fprintf(stderr, "Could not allocate nested sky-map workspace\n");
        free(map_nest);
        free(boundary_nest);
        free(inside90);
        return -1;
    }
    
    for(i = 0; i < Npix; i++)
    {
        ring2nest(Nside, i, &inest);
        map_nest[inest] = map[i];
        boundary_nest[inest] = 0.0;
        inside90[inest] = 0;
        boundary[i] = 0.0;
        region[i] = 0.0;
    }
    
    for(j = 0; j < k90; j++)
    {
        iring = (long)perm->data[Npix-1-j];
        ring2nest(Nside, iring, &inest);
        inside90[inest] = 1;
    }
    
    count = 0;
    for(inest = 0; inest < Npix; inest++)
    {
        if(!inside90[inest]) continue;
        
        skyhist_nested_neighbors(Nside, inest, neighbors);
        
        for(j = 0; j < 8; j++)
        {
            nneighbor = neighbors[j];
            if(nneighbor >= 0 && nneighbor < Npix && !inside90[nneighbor] && map_nest[nneighbor] <= level90 && boundary_nest[nneighbor] == 0.0)
            {
                boundary_nest[nneighbor] = 1.0;
                count++;
            }
        }
    }
    
    for(iring = 0; iring < Npix; iring++)
    {
        ring2nest(Nside, iring, &inest);
        boundary[iring] = boundary_nest[inest];
        region[iring] = (double)inside90[inest];
    }
    
    free(map_nest);
    free(boundary_nest);
    free(inside90);
    
    return count;
}

static void skyhist_nested_neighbors(long nside, long ipnest, long neighbors[8])
{
    static const int dx[8] = {-1, -1,  0,  1,  1,  1,  0, -1};
    static const int dy[8] = { 0,  1,  1,  1,  0, -1, -1, -1};
    long ix, iy, face, ixn, iyn;
    int i;
    
    skyhist_nest2xyf(nside, ipnest, &ix, &iy, &face);
    
    for(i = 0; i < 8; i++)
    {
        ixn = ix + dx[i];
        iyn = iy + dy[i];
        if(ixn >= 0 && ixn < nside && iyn >= 0 && iyn < nside)
        {
            neighbors[i] = skyhist_xyf2nest(nside, ixn, iyn, face);
        }
        else
        {
            neighbors[i] = skyhist_edge_neighbor_nest(nside, ix, iy, face, i);
        }
    }
}

static void skyhist_nest2xyf(long nside, long ipnest, long *ix, long *iy, long *face)
{
    long ipf, bit, shift;
    
    *face = ipnest/(nside*nside);
    ipf = ipnest - (*face)*nside*nside;
    *ix = 0;
    *iy = 0;
    
    for(bit = 1, shift = 0; bit < nside; bit <<= 1, shift++)
    {
        if(ipf & (1L << (2*shift))) *ix |= bit;
        if(ipf & (1L << (2*shift+1))) *iy |= bit;
    }
}

static long skyhist_xyf2nest(long nside, long ix, long iy, long face)
{
    long ipf, bit, shift;
    
    ipf = 0;
    for(bit = 1, shift = 0; bit < nside; bit <<= 1, shift++)
    {
        if(ix & bit) ipf |= (1L << (2*shift));
        if(iy & bit) ipf |= (1L << (2*shift+1));
    }
    
    return face*nside*nside + ipf;
}

static long skyhist_edge_neighbor_nest(long nside, long ix, long iy, long face, int dir)
{
    long N, r, face2, ix2, iy2;
    
    N = nside-1;
    face2 = -1;
    ix2 = -1;
    iy2 = -1;
    
    if(face < 4)
    {
        r = face;
        switch(dir)
        {
            case 0: face2 = face+4;       ix2 = N;    iy2 = iy;   break;
            case 1:
                if(ix == 0 && iy == N) return -1;
                if(ix == 0) { face2 = face+4;       ix2 = N;    iy2 = iy+1; }
                else        { face2 = skyhist_mod4(r+3);   ix2 = N;    iy2 = ix-1; }
                break;
            case 2: face2 = skyhist_mod4(r+3);   ix2 = N;    iy2 = ix;   break;
            case 3:
                if(ix == N && iy == N) { face2 = skyhist_mod4(r+2); ix2 = N;    iy2 = N;    }
                else if(iy == N)       { face2 = skyhist_mod4(r+3); ix2 = N;    iy2 = ix+1; }
                else                   { face2 = skyhist_mod4(r+1); ix2 = iy+1; iy2 = N;    }
                break;
            case 4: face2 = skyhist_mod4(r+1);   ix2 = iy;   iy2 = N;    break;
            case 5:
                if(ix == N && iy == 0) return -1;
                if(ix == N) { face2 = skyhist_mod4(r+1);   ix2 = iy-1; iy2 = N; }
                else        { face2 = 4+skyhist_mod4(r+1); ix2 = ix+1; iy2 = N; }
                break;
            case 6: face2 = 4+skyhist_mod4(r+1); ix2 = ix;   iy2 = N;    break;
            case 7:
                if(ix == 0 && iy == 0) { face2 = face+8;      ix2 = N;    iy2 = N;    }
                else if(ix == 0)       { face2 = face+4;      ix2 = N;    iy2 = iy-1; }
                else                   { face2 = 4+skyhist_mod4(r+1); ix2 = ix-1; iy2 = N;    }
                break;
        }
    }
    else if(face < 8)
    {
        r = face-4;
        switch(dir)
        {
            case 0: face2 = 8+skyhist_mod4(r+3);  ix2 = N;    iy2 = iy;   break;
            case 1:
                if(ix == 0 && iy == N) { face2 = 4+skyhist_mod4(r+3); ix2 = N;    iy2 = 0;    }
                else if(ix == 0)       { face2 = 8+skyhist_mod4(r+3); ix2 = N;    iy2 = iy+1; }
                else                   { face2 = skyhist_mod4(r+3);   ix2 = ix-1; iy2 = 0;    }
                break;
            case 2: face2 = skyhist_mod4(r+3);    ix2 = ix;   iy2 = 0;    break;
            case 3:
                if(ix == N && iy == N) return -1;
                if(iy == N) { face2 = skyhist_mod4(r+3); ix2 = ix+1; iy2 = 0;    }
                else        { face2 = r;         ix2 = 0;    iy2 = iy+1; }
                break;
            case 4: face2 = r;            ix2 = 0;    iy2 = iy;   break;
            case 5:
                if(ix == N && iy == 0) { face2 = 4+skyhist_mod4(r+1); ix2 = 0;    iy2 = N;    }
                else if(ix == N)       { face2 = r;           ix2 = 0;    iy2 = iy-1; }
                else                   { face2 = 8+r;         ix2 = ix+1; iy2 = N;    }
                break;
            case 6: face2 = 8+r;          ix2 = ix;   iy2 = N;    break;
            case 7:
                if(ix == 0 && iy == 0) return -1;
                if(ix == 0) { face2 = 8+skyhist_mod4(r+3); ix2 = N;    iy2 = iy-1; }
                else        { face2 = 8+r;         ix2 = ix-1; iy2 = N;    }
                break;
        }
    }
    else
    {
        r = face-8;
        switch(dir)
        {
            case 0: face2 = 8+skyhist_mod4(r+3);  ix2 = iy;   iy2 = 0;    break;
            case 1:
                if(ix == 0 && iy == N) return -1;
                if(ix == 0) { face2 = 8+skyhist_mod4(r+3); ix2 = iy+1; iy2 = 0; }
                else        { face2 = 4+r;         ix2 = ix-1; iy2 = 0; }
                break;
            case 2: face2 = 4+r;          ix2 = ix;   iy2 = 0;    break;
            case 3:
                if(ix == N && iy == N) { face2 = r;             ix2 = 0;    iy2 = 0;    }
                else if(iy == N)       { face2 = 4+r;           ix2 = ix+1; iy2 = 0;    }
                else                   { face2 = 4+skyhist_mod4(r+1);   ix2 = 0;    iy2 = iy+1; }
                break;
            case 4: face2 = 4+skyhist_mod4(r+1);  ix2 = 0;    iy2 = iy;   break;
            case 5:
                if(ix == N && iy == 0) return -1;
                if(ix == N) { face2 = 4+skyhist_mod4(r+1); ix2 = 0;    iy2 = iy-1; }
                else        { face2 = 8+skyhist_mod4(r+1); ix2 = 0;    iy2 = ix+1; }
                break;
            case 6: face2 = 8+skyhist_mod4(r+1);  ix2 = 0;    iy2 = ix;   break;
            case 7:
                if(ix == 0 && iy == 0) { face2 = 8+skyhist_mod4(r+2); ix2 = 0;    iy2 = 0;    }
                else if(ix == 0)       { face2 = 8+skyhist_mod4(r+3); ix2 = iy-1; iy2 = 0;    }
                else                   { face2 = 8+skyhist_mod4(r+1); ix2 = 0;    iy2 = ix-1; }
                break;
        }
    }
    
    if(face2 < 0) return -1;
    return skyhist_xyf2nest(nside, ix2, iy2, face2);
}

static long skyhist_mod4(long x)
{
    x %= 4;
    if(x < 0) x += 4;
    return x;
}


int *int_vector(int N)
{
    return malloc( (N+1) * sizeof(int) );
}

void free_int_vector(int *v)
{
    free(v);
}

int **int_matrix(int N, int M)
{
    int i;
    int **m = malloc( (N+1) * sizeof(int *));
    
    for(i=0; i<N+1; i++)
    {
        m[i] = malloc( (M+1) * sizeof(int));
    }
    
    return m;
}

void free_int_matrix(int **m, int N)
{
    int i;
    for(i=0; i<N+1; i++) free_int_vector(m[i]);
    free(m);
}

double *double_vector(int N)
{
    return malloc( (N+1) * sizeof(double) );
}

void free_double_vector(double *v)
{
    free(v);
}

double **double_matrix(int N, int M)
{
    int i;
    double **m = malloc( (N+1) * sizeof(double *));
    
    for(i=0; i<N+1; i++)
    {
        m[i] = malloc( (M+1) * sizeof(double));
    }
    
    return m;
}

void free_double_matrix(double **m, int N)
{
    int i;
    for(i=0; i<N+1; i++) free_double_vector(m[i]);
    free(m);
}

double ***double_tensor(int N, int M, int L)
{
    int i,j;
    
    double ***t = malloc( (N+1) * sizeof(double **));
    for(i=0; i<N+1; i++)
    {
        t[i] = malloc( (M+1) * sizeof(double *));
        for(j=0; j<M+1; j++)
        {
            t[i][j] = malloc( (L+1) * sizeof(double));
        }
    }
    
    return t;
}

void free_double_tensor(double ***t, int N, int M)
{
    int i;
    
    for(i=0; i<N+1; i++) free_double_matrix(t[i],M);
    
    free(t);
}

double ****double_quad(int N, int M, int L, int K)
{
    int i,j,k;
    
    double ****t = malloc( (N+1) * sizeof(double **));
    for(i=0; i<N+1; i++)
    {
        t[i] = malloc( (M+1) * sizeof(double *));
        for(j=0; j<M+1; j++)
        {
            t[i][j] = malloc( (L+1) * sizeof(double));
            for(k=0; k<L+1; k++)
            {
                       t[i][j][k] = malloc( (K+1) * sizeof(double));
            }
        }
    }
    
    return t;
}

void free_double_quad(double ****t, int N, int M, int L)
{
    int i;
    
    for(i=0; i<N+1; i++) free_double_tensor(t[i],M,L);
    
    free(t);
}

double fbegin(double *param)
{
    // [0] log(Mc) [1] log(Mt) [2] chi1 [3] chi2 [4] phi0 [5] tp [6] log(DL) [7] alpha [8] delta [9] psi [10] ciota
    
    double f, Mchirp, M, eta, dm, m1, m2, chi, chi1, chi2, tc, theta;
    double gamma_E=0.5772156649; //Euler's Constant-- shows up in 3PN term
    double PN1, PN15, PN2, PN25, PN3, PN35;
    double theta2, theta3, theta4, theta5, theta6, theta7;
    
    m1 = param[0]*TSUN;
    m2 = param[1]*TSUN;
    M = m1+m2;
    eta = m1*m2/(M*M);
    chi1 = param[2];
    chi2 = param[3];
    chi = (m1*chi1+m2*chi2)/M;
    tc = param[5];
    
    theta = pow(eta*tc/(5.0*M),-1.0/8.0);
    theta2 = theta*theta;
    theta3 = theta2*theta;
    theta4 = theta2*theta2;
    theta5 = theta2*theta3;
    theta6 = theta3*theta3;
    
    
    PN1 = (11.0/32.0*eta+743.0/2688.0)*theta2;
    PN15 = -3.0*PI/10.0*theta3 + (1.0/160.0)*(113.0*chi-38.0*eta*(chi1+chi2))*theta3;
    PN2 = (1855099.0/14450688.0+56975.0/258048.0*eta+371.0/2048.0*eta*eta)*theta4 + (1.0/14450688.0)*(-3386880.0*chi*chi+1512.0*chi1*chi2)*theta4;
    PN25 = -(7729.0/21504.0-13.0/256.0*eta)*PI*theta5;
    PN3 = (-720817631400877.0/288412611379200.0+53.0/200.0*PI*PI+107.0/280.0*gamma_E+(25302017977.0/4161798144.0-451.0/2048.0*PI*PI)*eta-30913.0/1835008.0*eta*eta+235925.0/1769472.0*eta*eta*eta + 107.0/280.0*log(2.0*theta))*theta6;

    
    f = theta3/(8.0*M*PI)*(1.0 + PN1 + PN15 + PN2 + PN25 + PN3);

    return(f);
    
}

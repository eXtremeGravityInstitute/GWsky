# GWsky
Codes to produce gravitational wave sky localization maps

This package produces sky maps for a network of gravitational wave detectors. The analysis is focused on binary neutron star mergers, but can easily be adapted for any other type of compact binary merger. The code is set up to include any combination of the LIGO Hanford, Livingston and India observatories, and the Virgo observatory. The default is to use noise curves for the O5 run. It uses the Aplus design sensitivity for Hanford and Livingston, the best O4 sensitivity for India and the high noise (conservative) O5 projection for Virgo.

The code uses IMRPhenomD waveforms and fast Bayesian inference. One caveat, the analysis holds the intrinsic parameters - the masses and spins - fixed, while performing a full MCMC over the sky location, distance, arrival time, overall phase, polarization angle and inclination angle. The rationale is that the covariance between the intrinsic and extrinsic parameters is small, and holding the intrinsic parameter fixed allows for an extremely fast likelihood evaluation that avoided repeated calls to the waveform generator - we simply re-project the reference waveform. The analysis is based on the sky mapping subroutines from the QuickCBC analysis package Phys.Rev.D 103 (2021) 10, 104057 • e-Print: 2101.01188.

The code is written in C and uses the GSL and c-Healpix libraries. The compile line for a home-brew install of GSL is given in the code header. 

The command 

./sky_map 0

will tell the code to read in the injection file, injection.dat, and produce a sky map for this source. The format of the file is mass1, mass2, chi1, chi2, phase, merger time, distance, RA, DEC, polarization, cos(inclination). All angles are in radians. The masses are in solar mass units. The distance is in Mpc. The merger time is in seconds, but gets overwritten inside the code to put it 4 seconds before the end of the default 1024 second analysis window.

The command

./sky_map 100

will generate a catalog of 100 detectable BNS mergers (SNR >= 8). The code randomly draws sources uniformly in co-moving volume, with random orientations and masses that follow a Gaussian distribution with mean 1.4 and standard deviation 0.2. The dimensionless spins are uniform in the range 0 to 0.1. The maximum distance for then uniform in volume draw is fixed by the overall detector sensitivity, which is computed first and reported. The BNS range uses the standard LIGO definition and is computed using a Monte Carlo integration that is very fast. The code also computes the number of detections per year using the GWTC-5 BNS rate as a reference. The reference rate range can be changed in the header file.

The code produces fairly verbose output as it runs, including the SNR of each source, the 50%n and 90% credible sky areas, the MCMC proposal acceptances etc. Don't worry if the Ring and Antipodal proposals have low acceptance rates. With a 4 detector network the sky maps are usually single regions, without the big bananas of a Hanford-Livingston map. 

All of the output goes into the skymaps sub-directory. This includes Heaplix maps for each source (not yet rendered into pngs), boundary maps for the 90% regions, summaries for each source etc. In addition, when run in catalog mode, the code produces histograms of the 50% and 90% credible sky areas, and histograms of the injected distances and SNRs. These are saved as skymaps/population_histograms.png.

The code has several default settings, such as the sample rate and observation time. Do not reduce the sample rate if you want reliable sky maps. The observation time can be reduced for higher mass systems such as NS-BH binaries. The population model would obviously have to be changed for NS-BH or BH-BH binaries.

The code can be run on any combination of the 4 detectors. See lines 167-169 for instructions on how to do this. To use different sensitivity curves you would need to provide the ASD files and edit lines 223-229.

The code does not render all the sky maps. But you can generate the sky maps and 90% credible boundaries for any source using the script skyplot.sh. For example, ./skyplot.sh 2 will produce the sky map for source 2 in the catalog. To get a sense of what the sky maps look like for a collection of sources, execute ./plot_boundary_sample.py 50, which will produce the file skymaps/sky90_boundary_sample_50.png for 50 sources. This code skips the rare sources that have large sky areas because they are not interesting, and because they make the plot hard to read.

For the default 4 detector network and reference O5 sensitivities, the network BNS range is 556 Mpc. Using just the Hanford or Livingston detectors, which are set at the Aplus design sensitivity, the code reports a BNS range of 337 Mpc. This is a little higher than the 325 Mpc quoted in the latest LIGO documentation for O5, but the waveform models are a little different, so the agreement is reasonable.


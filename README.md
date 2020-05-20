# BAYES-LOSVD

Non-parametric recovery of the LOSVD in galaxies

*Usage*

- Compile all the Stan codes\n

python bayes-bayes_compile_codes.py

FOR TEST DATA

  - Create test data

  python create_testdata.py -c ../config_files/default_testdata.conf

  - Preprocess data

  python bayes-losvd_preproc_data.py -c ../config_files/default_testcases.conf

FOR REAL DATA

  python bayes-losvd_preproc_data.py -c ../config_files/default.conf

- Run code

python bayes-losvd_run.py -f \<preproc HDF5 file\> -b \<bin number or list\> -t <type of fit\> -v \<verbose\> -p \<flag to save diagnostic plots\> -s \<flag to save chains\>

- Inspect results

python bayes-losvd_inspect_fits.py -r \<runname in config file\>+"-"+fit_type -b \<bin number or list\> -s \<flag to save plot\>

*Notes*

You will see that results are stored differently in the results directory.

Data is sotred in HDF5 files with a different structure that allows to collect all the info of the individual bins into a single file.

If required, chains can be saved in NETCDF format for further processing with Arviz



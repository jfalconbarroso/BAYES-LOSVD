# BAYES-LOSVD

Non-parametric recovery of the LOSVD in galaxies

*Usage*

- Compile all the Stan codes\n

python bayes-bayes_compile_codes.py

- Create test data

python create_testdata.py -c ../config_files/default_testdata.conf

- Preprocess data

python bayes-losvd_preproc_data.py -c ../config_files/default_testcases.conf
python bayes-losvd_preproc_data.py -c ../config_files/default.conf

- Run code

python bayes-losvd_run.py -r <runname in config file> -b <bin number or list> -v <verbose> -p <flag to save diagnostic plots> -s <flag to save chains>

- Inspect results

python bayes-losvd_inspect_fits.py -r <runname in config file> -b <bin number or list> -s <flag to save plot>

*Notes*

You will see that results are stored differently in the results directory
Data is sotred in HDF5 files with a different structure that allows to collect all the info of the individual bins into a single file
The analysis scripts contains the scripts to generate the plots I've been sending you. It takes the results I've generated in the machine 'denso' at the IAC. You need to chage directory accordingly.



# BAYES-LOSVD

Example for one case

*Usage*

- Compile all the Stan codes\n

python bayes-bayes_compile_codes.py

- Preprocess data

  python bayes-losvd_preproc_data.py -c ../config_files/default.conf

- Run code

python bayes-losvd_run.py -f ../preproc/NGC4550_SAURON.hdf5 -b 0,169,189 -t <type of fit\> -v \<verbose\> -p \<flag to save diagnostic plots\> -s \<flag to save chains\>

-t \<type of fit\>: [S0: Free, S1:RW, AX: AR of order X, BX: B-splines of order X, G0: GH free]
-v \<verbose\> [0: no, 1:yes]
-p \<flag to same plots\> [0: no, 1:yes]
-s \<flag to save chains\> [0: no, 1:yes]

- Inspect results interactively

python bayes-losvd_inspect_fits.py -r NGC4550_SAURON-G0 -b 0 -s \<flag to save plot\>

For further analysis plots go to 'analysis_scripts'

python plot_data_specfits.py
[Check the call inside the script at the bottom]





# BAYES-LOSVD

*Usage*

Move to the 'scripts' directory. All scripts have to be run from there.

- Compile all the Stan codes
 
  python bayes-bayes_compile_codes.py

- Preprocess data

  python bayes-losvd_preproc_data.py -c ../config_files/IC0719.conf

- Run code

  python bayes-losvd_run.py -f ../preproc/IC0719-test_blue.hdf5 -b \<bin list\> -t <type of fit\> -v \<verbose\> -p \<flag to save diagnostic plots\> -s \<flag to save chains\>

  -b \<bin list\>: [all, odd, even, bin number]

  -t \<type of fit\>: [S0: Free, S1:RW, AX: AR of order X, BX: B-splines of order X, G0: GH free]

  -v \<verbose\> [0: no, 1:yes]

  -p \<flag to same plots\> [0: no, 1:yes]

  -s \<flag to save chains\> [0: no, 1:yes]

  EXAMPLE: 

    python bayes-losvd_run.py -f ../preproc/IC0719-test_blue.hdf5 -b all -t A2 -v 1 -p 1 -s 0  

    python bayes-losvd_run.py -f ../preproc/IC0719-test_red.hdf5  -b all -t A2 -v 1 -p 1 -s 0  

- Inspect results interactively

  python bayes-losvd_inspect_fits.py -r IC0719-test_blue-A2 -b 0 -s 0
  
  python bayes-losvd_inspect_fits.py -r IC0719-test_red-A2 -b 0 -s 0

For further analysis plots go to 'analysis_scripts'

  python plot_data_specfits.py  [Check the call inside the script at the bottom]





//=============================================================================
functions{

  // Function to build splines
  // From: https://mc-stan.org/users/documentation/case-studies/splines_in_stan.html
  vector build_b_spline(real[] t, real[] ext_knots, int ind, int order);
  vector build_b_spline(real[] t, real[] ext_knots, int ind, int order) {
     // INPUTS:
     //    t:          the points at which the b_spline is calculated
     //    ext_knots:  the set of extended knots
     //    ind:        the index of the b_spline
     //    order:      the order of the b-spline
     vector[size(t)] b_spline;
     vector[size(t)] w1 = rep_vector(0, size(t));
     vector[size(t)] w2 = rep_vector(0, size(t));
     if (order==1)
       for (i in 1:size(t)) // B-splines of order 1 are piece-wise constant
         b_spline[i] = (ext_knots[ind] <= t[i]) && (t[i] < ext_knots[ind+1]);
     else {
       if (ext_knots[ind] != ext_knots[ind+order-1])
         w1 = (to_vector(t) - rep_vector(ext_knots[ind], size(t))) /
              (ext_knots[ind+order-1] - ext_knots[ind]);
       if (ext_knots[ind+1] != ext_knots[ind+order])
         w2 = 1 - (to_vector(t) - rep_vector(ext_knots[ind+1], size(t))) /
                  (ext_knots[ind+order] - ext_knots[ind+1]);
       // Calculating the B-spline recursively as linear interpolation of two lower-order splines
       b_spline = w1 .* build_b_spline(t, ext_knots, ind, order-1) +
                  w2 .* build_b_spline(t, ext_knots, ind+1, order-1);
     }
     return b_spline;

  }

  // Function that convolves an input spectrum with a given kernel
  // (Adapted from https://discourse.mc-stan.org/t/discrete-convolution-by-direct-summation-can-this-be-made-more-efficient/969/3) 
  vector convolve_data(vector spec, vector kernel, int npix_temp, int nk){

     row_vector[nk]          kernelp  = kernel';
     vector[npix_temp-nk+1]  out_spec = rep_vector(1.0,npix_temp-nk+1);

     for (i in 1:(npix_temp-nk+1)){
        out_spec[i] = kernelp * spec[i:(i+nk-1)];
     }    

     return out_spec;   

  }  

  // Function to create a fiducial vector from 1 to npix 
  vector create_vector(int npix){

     vector[npix] vect;

     for (i in 1:npix){
       vect[i] = i;
     }    

     return vect;
  }    

  // Function to scale the fiducial vector in the range [-1,1]
  vector scale_vector(vector vect, int npix){

     real vectorMax = max(vect);
     real vectorMin = min(vect);
     real minRange = -1.0;
     real maxRange =  1.0;
     real scale_factor0 = ((minRange * vectorMax)-(maxRange * vectorMin)) / (vectorMax - vectorMin);
     real scale_factor1 = (maxRange - minRange) / (vectorMax - vectorMin);
     vector[npix] out_data = scale_factor0 + vect * scale_factor1;

     return out_data;
  }

  // Function that creates the Legendre polynomials up to order k
  // The Legendre polynomial P(n,x) can be defined by:
  //      P(0,x) = 1
  //      P(1,x) = x
  //      P(n,x) = [(2*n-1) * x * P(n-1,x) - (n-1) * P(n-2,x)]/n
  // NOTE: careful! the indexing of Stan arrays starts at 1, not 0!
  matrix legendre(vector x, int k, int npix){

      matrix[npix,k+1] plgndr;

      // P(0) 
      plgndr[:,1] = rep_vector(1.0,npix);

      // P(n>=1)
      if (k > 0){
          plgndr[:,2] = x;
          for (n in 2:k){  
            plgndr[:,n+1] = ((2.0*n-1.0) * x .* plgndr[:,n] - plgndr[:,n-1] * (n-1)) / n;
          }  
      }

      return plgndr;
  }      

}
//=============================================================================
data {

  int<lower=1> npix_obs;                     // Number of pixels of input spectrum
  int<lower=1> ntemp;                        // Number of PC components
  int<lower=1> npix_temp;                    // Number of pixels of each PC components
  int<lower=1> nvel;                         // Number of pixels of the LOSVD
  int<lower=1> nmask;                        // Number of pixels of the data mask
  int<lower=1> mask[nmask];                  // Mask with the data pixels to be fitted
  int<lower=0> porder;                       // Polynomial order to be used
  //-------------------------
  vector[npix_obs]            spec_obs;      // Array with observed spectrum
  vector<lower=0.0>[npix_obs] sigma_obs;     // Array with error espectrum
  matrix[npix_temp,ntemp]     templates;     // Array with PC components spectra
  vector[npix_temp]           mean_template; // Array with mean template of the PCA decomposion
  vector[nvel] xvel;                         // Velocity vector to be used as knots  
  //-------------------------
  int<lower=0> spline_order;                        // the degree of spline (is equal to B-splines order - 1)
  vector[nmask]               spec_masked;

}
//=============================================================================
transformed data{

  // Number of knots to be used in the B-splines
  int<lower=3> num_knots = nvel; 

  // Building the Legendre polynomials
  vector[npix_obs]          vect     = create_vector(npix_obs);
  vector[npix_obs]          scl_vect = scale_vector(vect,npix_obs);
  matrix[npix_obs,porder+1] leg_pols = legendre(scl_vect,porder,npix_obs);

  // Building the B-Splines
  int                         num_basis = num_knots + spline_order - 1; // total number of B-splines
  vector[num_knots]           vect2 = create_vector(num_knots);
  vector[num_knots]           knots = scale_vector(vect2,num_knots)*max(xvel);
  matrix[num_basis, nvel]     B;             // matrix of B-splines
  matrix[nvel,num_basis]      B_transposed;  // matrix of B-splines transposed
  vector[spline_order + num_knots]   ext_knots_temp;
  vector[2*spline_order + num_knots] ext_knots; // set of extended knots

  ext_knots_temp = append_row(rep_vector(knots[1], spline_order), knots);
  ext_knots      = append_row(ext_knots_temp, rep_vector(knots[num_knots], spline_order));
  for (ind in 1:num_basis)
    B[ind,:] = to_row_vector(build_b_spline(to_array_1d(xvel), to_array_1d(ext_knots), ind, spline_order + 1));
  B[num_knots + spline_order - 1, nvel] = 1;
  B_transposed = B';

}    
//=============================================================================
parameters {

  // Parameters for main model
  vector<lower=-2.0,upper=2.0>[ntemp]    weights;  // Weights for each PC component 
  vector<lower=-2.0,upper=2.0>[porder+1] coefs;    // Coefficients of the Legendre polynomials
  simplex[num_basis]                     a;        // B-splines coefficients
  real<lower=0.0>                        sigma;    // Smoothing for RW prior           
 
}
//=============================================================================
transformed parameters {

  vector<lower=0.0>[nvel] losvd_ = B_transposed*a;       // B-splines LOSVD
  simplex[nvel]           losvd  = losvd_ / sum(losvd_);  // Normalised LOSVD                   

}
//=============================================================================
model {

  // Defining model  
  vector[npix_temp] spec       = mean_template + templates * weights;       
  vector[npix_obs]  conv_spec  = convolve_data(spec,losvd,npix_temp,nvel);
  vector[npix_obs]  model_spec = leg_pols * coefs + conv_spec;

  // Weakly informative priors on PCA weights, Leg. and B-splines coefficients
  weights ~ normal(0.0,1.0);
  coefs   ~ normal(0.0,1.0);
  sigma   ~ normal(0.0,1.0);
  a[1]    ~ normal(0.0,sigma);
  a[2:num_basis-1] ~ normal(a[1:(num_basis-2)],sigma);

  // Inference
  spec_obs[mask] ~ normal(model_spec[mask],sigma_obs[mask]);

}
//=============================================================================
generated quantities {

  vector[npix_temp] spec      = mean_template + templates * weights;       
  vector[npix_obs]  conv_spec = convolve_data(spec,losvd,npix_temp,nvel);
  vector[npix_obs]  poly      = legendre(scl_vect,porder,npix_obs) * coefs;
  vector[npix_obs]  bestfit   = poly + conv_spec;
  vector[nmask]     spec_pred;
  vector[nmask]     log_likelihood;
  for (i in 1:nmask){
    log_likelihood[i] = normal_lpdf(spec_obs[mask[i]] | bestfit[mask[i]], sigma_obs[mask[i]]);
    spec_pred[i]      = normal_rng(bestfit[mask[i]], sigma_obs[mask[i]]);
  }   

}
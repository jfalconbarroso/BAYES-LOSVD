//=============================================================================
functions{

 // Function that convolves an input spectrum with a given kernel
  // (Adapted from https://discourse.mc-stan.org/t/discrete-convolution-by-direct-summation-can-this-be-made-more-efficient/969/3) 
  vector convolve_data(vector spec, vector kernel, int npix, int nk){
      
     row_vector[nk] kernelp  = kernel';
     vector[npix]   out_spec = rep_vector(1.0,npix);
   
     for (i in 1:npix){
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
  
  // Function to scale the fiducial vector above in the range [-1,1]
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
  
 

  // Function that creates a GH LOSVD expansion given vel, sigma, h1, h2, h3, h4
  vector losvd_func(vector xvel, real vel, real sigma, real h1, real h2, real h3, real h4, int nvel){
      
    vector[nvel] w     = (xvel - vel)/sigma;
    vector[nvel] w2    = square(w);
    vector[nvel] losvd = exp(-0.5*w2)/(sqrt(2.0*pi())*sigma);
    vector[nvel] poly;
    
    poly = 1.0 + h1*sqrt(2.0)*w + (h2/sqrt(2.0))*(2*w2-1) +(h3/sqrt(3.0))*(w .* (2.0*w2-3.0)) + (h4/sqrt(24))*(w2 .* (4.0*w2-12.0)+3.0);
    losvd .*= poly;
    
    for (i in 1:nvel){
    	if(losvd[i] <= 0){
	    losvd[i] = 1e-10;
	}
    }    
    losvd = losvd/sum(losvd);
        
    return losvd;  
    
  }



        
}
//=============================================================================

data {
    
  int<lower=1> npix_obs;                 // Number of pixels of input spectrum
  int<lower=1> ntemp;                    // Number of PC components
  int<lower=1> npix_temp;                // Number of pixels of each PC components
  int<lower=1> nmask;                    // Number of pixels of the mask
  int<lower=1> mask[nmask];              // Mask with pixels to be fitted
  int<lower=0> porder;                   // Polynomial order to be used
  real<lower=0> velscale;                // Velocity scale
  int<lower=1> nvel;                     // Number of pixels of the LOSVD
  vector[nvel]  xvel;                    // x array for the LOSVD

  //-------------------------
  matrix[npix_temp,ntemp]     templates;     // Array with PC components spectra
  vector[npix_temp]           mean_template; // Array with mean template of the PCA decomposion
  vector[npix_obs]            spec_obs;      // Array with observed spectrum
  vector<lower=0.0>[npix_obs] sigma_obs;     // Array with error espectrum
  vector[nmask]               spec_masked;
    
}

//=============================================================================
transformed data{
 
  vector[npix_obs]          vect     = create_vector(npix_obs);
  vector[npix_obs]          scl_vect = scale_vector(vect,npix_obs);
  matrix[npix_obs,porder+1] leg_pols = legendre(scl_vect,porder,npix_obs);

}   
//=============================================================================
parameters {
  
  // GH parameters
  real <lower=min(xvel), upper=max(xvel)> vel ;
  real <lower=velscale/2., upper= 500.0> sigma ;
  real <lower=-1.0,upper=1.0> h1 ;			
  real <lower=-1.0,upper=1.0> h2 ;			
  real <lower=-1.0,upper=1.0> h3 ;			
  real <lower=-1.0,upper=1.0> h4 ;			

  vector<lower=-2.0,upper=2.0>[ntemp] weights;   // Weights for each PC component
  vector<lower=-2.0,upper=2.0>[porder+1] coefs;  // Coefficients of the polynomials
  
}
//=============================================================================
transformed parameters {

  vector[nvel] losvd     = losvd_func(xvel,vel, sigma, h1, h2, h3, h4, nvel);

}
//=============================================================================
model {
        
  // Defining model
  vector[npix_temp] spec       = mean_template + templates * weights;       
  vector[npix_obs]  conv_spec  = convolve_data(spec,losvd,npix_obs,nvel);
  vector[npix_obs]  model_spec = leg_pols * coefs + conv_spec;

  // Priors on PCA weights and polynomial coeffs
  coefs   ~ normal(0.0,1.0);
  weights ~ normal(0.0,1.0);

  // Priors on GH parameters
  h1 ~ normal(0.0, 0.2);
  h2 ~ normal(0.0, 0.2);
  h3 ~ normal(0.0, 0.2);
  h4 ~ normal(0.0, 0.2);
  
   // Inference
  spec_obs[mask] ~ normal(model_spec[mask],sigma_obs[mask]);

}

//=============================================================================
generated quantities {

  vector[npix_temp] spec      = mean_template + templates * weights;
  vector[npix_obs]  conv_spec = convolve_data(spec,losvd,npix_obs,nvel);
  vector[npix_obs]  poly      = leg_pols * coefs;
  vector[npix_obs]  bestfit   = poly + conv_spec;
  vector[nmask]     spec_pred;
  vector[nmask]     log_likelihood;
  for (i in 1:nmask){
    log_likelihood[i] = normal_lpdf(spec_obs[mask[i]] | bestfit[mask[i]], sigma_obs[mask[i]]);
    spec_pred[i]      = normal_rng(bestfit[mask[i]], sigma_obs[mask[i]]);
  }   

}


//=============================================================================
functions{
       
  // Function that creates a GH LOSVD expansion given gamma, vel, sigma, h3, h4
  vector losvd_func(vector xvel, real gamma, real vel, real sigma, real h3, real h4, int nvel){
      
    vector[nvel] w     = (xvel - vel)/sigma;
    vector[nvel] w2    = square(w);
    vector[nvel] losvd = gamma*exp(-0.5*w2)/(sqrt(2.0*pi())*sigma);
    vector[nvel] poly;
    
    poly = 1.0 + (h3/sqrt(3.0))*(w .* (2.0*w2-3.0)) + (h4/sqrt(24.0))*(w2 .* (4.0*w2-12.0)+3.0);
    losvd .*= poly;
        
    return losvd;  
    
  }    
     
}
//=============================================================================
data {

  int<lower=1>    nvel;        // Number of velocity bins
  vector[nvel]    xvel;        // Vector with input velocity bins
  vector[nvel]    losvd_obs;   // Vector with input LOSVD
  vector<lower=0.0>[nvel] sigma_losvd; // Uncertainty of the LOSVD
    
}
//=============================================================================
parameters {
  
  real<lower=0.0> offset;  
  real<lower=0.0> gamma;
  real<lower=-1000.0,upper=1000.0> vel;
  real<lower=0.0,upper=500.0> sigma;
  real<lower=-0.25,upper=0.25> h3;
  real<lower=-0.25,upper=0.25> h4;
}
//=============================================================================
model {
        
  // Defining model  
  vector[nvel] losvd_mod = offset + losvd_func(xvel, gamma, vel, sigma, h3, h4, nvel);
      
  // Weakly informative Priors
  offset ~ normal(0.0,0.2);
  gamma  ~ normal(0.0,1000.0);
  vel    ~ normal(0.0,200.0);
  sigma  ~ normal(0.0,200.0);
  h3     ~ normal(0.0,0.1);
  h4     ~ normal(0.0,0.1);
  
  // Inference
  losvd_obs ~ normal(losvd_mod,sigma_losvd);

}
//=============================================================================
generated quantities{
 
  vector[nvel] losvd_mod = offset + losvd_func(xvel, gamma, vel, sigma, h3, h4, nvel);

}    

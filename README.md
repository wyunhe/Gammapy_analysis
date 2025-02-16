# Gammapy_analysis

This repository provides the code and notebooks used for my Master's thesis work. 

## Contents

1. Pulsar Data Processing
- `Addphase.ipynb` – Adds pulsar phase information to DL3 FITS files by matching event IDs from melibea ROOT files  
- `Data_reduction.ipynb` – Performs data reduction for pulsar DL3 data

2. Flux Point Estimation
   
**MCMC technique** with **Tikhonov Regularization** for flux point estimation in Gammapy  
  - Implementation: `mcmc_reg.py`  
  - Example use: `Example_flux_estimation_mcmc_reg.ipynb`  
  - Method details: `Details_and_performance.ipynb`  
  - `find_opt.py` : helps to determine the optimal regularization depth  
&nbsp;

## Flux Point Estimation: Stepped Power Law Model + MCMC Sampling with Tikhonov Regularization

- It is based on the stepped power law model, proposed to address two main **issues** in Gammapy's standard `FluxPointEstimator`:  
  a) Its dependence on a predefined reference spectrum, which may introduce unwanted correlations.  
  b) Its operation in the reconstructed energy space, rather than the true energy space, which can lead to
unreliable results on the edge of the underlying spectrum

- Directly fitting the model parameters resulted in large uncertainties and correlations between adjacent energy bins, due to the ill-posed nature of the problem. Therefore, the **Tikhonov regularization term** was introduced to obtain reliable results


It provides tools for:  

This repository for Master thesis work, it provides:
- Addphase.ipynb : to add pulsar phase to DL3 FITS files
- Data_reduction.ipynb: data reduction of pulsar DL3 data
- MCMC technique with Tikhonov Regularization for flux point estimation in Gammapy
  - implementation : mcmc_reg.py
  - Example use : Example_flux_estimation_mcmc_reg.ipynb
  - method details : Details_and_performance.ipynb
  - find_opt.py help to determine the optimal regularization depth

# Flux Point Estimation: Stepped power law model + MCMC sampling with Tikhonov Regularization 

- It is based on the stepped power law model, proposed to address the two main issues with Gammapy's standard FluxPointEstimator:
a) its dependence on a predefined reference spectrum, which may introduce unwanted correlations;
b) its operation in the reconstructed energy space, rather than the true energy space, which can lead to
unreliable results on the edge of the underlying spectrum

- Directly fitting the model parameters resulted in large uncertainties and correlations between adjacent energy bins, due to the ill-posed nature of the problem. Therefore, the Tikhonov regularization term was introduced to obtain reliable results



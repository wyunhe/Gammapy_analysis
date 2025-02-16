"""
MCMC sampling with Tikhonov regularization for flux point estimation in Gammapy

The flux point estimation is based on the stepped power law model, which addresses issues in Gammapy’s standard estimator FluxPointEstimator

The Tikhonov regularization term is introduced to solve the ill-posed nature of the problem, which limits the performance of direct fitting of the stepped power law model.

Usage:
    For a practical example, see Example_flux_estimation_mcmc_reg.ipynb

Further Details:
    For more details, see Details_and_performance.ipynb

"""

"""MCMC sampling helper functions using ``emcee``."""
import logging
import numpy as np
import gammapy
from IPython.display import display, Math
from astropy import units as u
from gammapy.modeling.models import SpectralModel
from gammapy.modeling import Parameter, Parameters

log = logging.getLogger(__name__)

# the Tikhonov regularization term is added in the lnprob_with_regularization()

############################  the stepped power law model   #############################

class PiecewiseConstSpectralModel(SpectralModel):
    """
    Piecewise constant spectral correction model with a free constant 
    norm parameter for each specified energy bin.

    Parameters
    ----------
    energy_edges : `~astropy.units.Quantity`
        Array of energy bin edges, defining the boundaries of each bin.
    norms : list of float, optional
        List of normalization values for each energy bin. If not provided, 
        defaults to 1.0 for each bin.
    """
    
    tag = ["PiecewiseConstSpectralModel", "piecewise-const"]
    
    def __init__(self, energy_edges, norms=None):
        self._energy_edges = energy_edges

        if norms is None:
            norms = np.ones(len(energy_edges) - 1)

        if len(norms) != len(energy_edges) - 1:
            raise ValueError("The number of norms must be one less than the number of energy edges.")
        if len(norms) < 1:
            raise ValueError("Input arrays must contain at least 2 edges to define 1 bin.")

        # Define normalization parameters for each energy bin
        parameters_list = []
        if not isinstance(norms[0], Parameter):
            parameters_list += [
                Parameter(f"norm_{k}", norm) for k, norm in enumerate(norms)
            ]
        else:
            parameters_list += norms
            
        self.default_parameters = Parameters(parameters_list)
        super().__init__()

    @property
    def energy_edges(self):
        """Energy bin edges."""
        return self._energy_edges

    @property
    def norms(self):
        """Norm values"""
        return u.Quantity([p.value for p in self.parameters])

    def evaluate(self, energy, **kwargs):
        """
        Evaluate the piecewise constant model at a given energy.
        Each bin applies a constant correction within its edges.
        """
        e_edges = self.energy_edges.to(energy.unit).value
        v_norms = self.norms

        # Identify which bin each energy value falls into
        # e_edges[idx-1] <= e < e_edges[idx]
        idx = np.searchsorted(e_edges, energy.value, side="right") - 1
        idx = np.clip(idx, 0, len(v_norms)-1)
        
        return v_norms[idx]

    def to_dict(self, full_output=False):
        data = super().to_dict(full_output=full_output)
        data["spectral"]["energy"] = {
            "data": self.energy.data.tolist(),
            "unit": str(self.energy.unit),
        }
        return data

    @classmethod
    def from_dict(cls, data, **kwargs):
        """Create model from dictionary."""
        data = data["spectral"]
        energy = u.Quantity(data["energy"]["data"], data["energy"]["unit"])
        parameters = Parameters.from_dict(data["parameters"])

    @classmethod
    def from_parameters(cls, parameters, **kwargs):
        """Create model from parameters."""
        return cls(norms=parameters, **kwargs)


##########################################################################################

def compute_reg(B_arr, reg_depth=1):
    """ Calculate the Tikhonov regularization term for a given distribution of parameters (B_arr)"""
    reg_B = 0.0
    mask = np.nonzero(B_arr)
    B_arr = B_arr[mask]
    
    for j in range(1, len(B_arr) - 1):
        term1 = 2 * (B_arr[j+1] - B_arr[j]) / (B_arr[j+1] + B_arr[j])
        term2 = 2 * (B_arr[j] - B_arr[j-1]) / (B_arr[j] + B_arr[j-1])
        reg_B += ((term1 - term2)) ** 2
    
    return reg_depth * reg_B


def get_npred_Etrue(dataset):
    """ Compute the predicted count distribution with respect to the E_true axis. (which is the B_arr in the compute_reg())
    
    This function extracts the model from the dataset, computes its flux, applies the exposure (* Aeff *livetime), and 
    returns the predicted count distribution (B_arr) with respect to the E_true axis.
    """
    model_name = dataset.models.names[0]
    eval = dataset.evaluators[model_name]

    flux_val = eval.compute_flux_spectral()
    flux = gammapy.maps.Map.from_geom(geom=eval.geom, data=flux_val.value, unit=flux_val.unit)
    npred = eval.apply_exposure(flux)
    B_arr = npred.data[:,0,0]

    return B_arr


def get_npred_err(dataset):
    """ Compute the error of the predicted count distribution with respect to the E_true axis. 
    
    """
    model_name = dataset.models.names[0]
    eval = dataset.evaluators[model_name]

    energy_true = eval.geom.axes['energy_true'].edges

    flux_val, flux_err_val = eval.model.spectral_model.integral_error(energy_true[:-1], energy_true[1:])
    flux_err = gammapy.maps.Map.from_geom(geom=eval.geom, data=flux_err_val.value, unit=flux_err_val.unit)
    
    #npred = eval.apply_exposure(flux)
    npred_err = eval.apply_exposure(flux_err)
    
    return npred_err.data[:,0,0]


def tr_cov_npred(dataset):
    """ Compute the trace of the covariance matrix of predicted counts with respect to the E_true axis.
    """
    npred_err = get_npred_err(dataset)
    return sum(npred_err**2)


def tr_cov_data(dataset):
    """  Compute the trace of the covariance matrix of measured excess counts.
    """
    data_err = dataset._counts_statistic.error[:,0,0]
    return sum(data_err**2)


def set_mcmc_results(sampler, dataset, nburn, print=False):
    """  Get median, lower and upper error of the parameters (16th and 84th percentile), used for the plot;
        Reset the spectral model with errors from diag(cov(sampler))
    """
    free_param = dataset.models.parameters.free_parameters
    flat_samples = sampler.get_chain(discard=nburn, flat=True)
    flat_samples_rescale = np.zeros_like(flat_samples)

    median = np.zeros(len(free_param))
    err_minus = np.zeros(len(free_param))
    err_plus = np.zeros(len(free_param))
    
    for i, par in enumerate(free_param):
        flat_samples_rescale[:, i] = flat_samples[:, i] * par.scale
        #mean[i] = np.mean(flat_samples_rescale[:, i])
        mcmc = np.percentile(flat_samples_rescale[:, i], [16, 50, 84])
        median[i] = mcmc[1]
        sig = np.diff(mcmc)
        err_minus[i] = sig[0]
        err_plus[i] = sig[1]

        txt = r"\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}} \;"
        txt = txt.format(median[i], sig[0], sig[1], par.name)
        log.info(txt)
        if print:
            display(Math(txt))

    cov_param = np.cov(flat_samples_rescale.T)
    std = np.sqrt(np.diag(cov_param))
    # reset the spectral model
    for i, par in enumerate(free_param):
        par.value = median[i]
        par.error = std[i]
            
    return median, err_minus, err_plus


def check_regularization(sampler, dataset, nburn):
    """ Print the values of the trace of two covariance matrix: 
        one for measured excess counts; one for predicted counts with respect to the E_true axis
    """
    set_mcmc_results(sampler, dataset, nburn, print=False)

    tr_data = tr_cov_data(dataset)
    log.info(f"tr_cov_data = {tr_data}")
    
    tr_npred = tr_cov_npred(dataset)
    log.info(f"tr_cov_npred = {tr_npred}")
    
    return tr_data, tr_npred
    


######################### likelihood with Tikhonov regularization #############################

def uniform_prior(value, umin, umax):
    """Uniform prior distribution."""
    if umin <= value <= umax:
        return 0.0
    else:
        return -np.inf


def normal_prior(value, mean, sigma):
    """Normal prior distribution."""
    return -0.5 * (2 * np.pi * sigma) - (value - mean) ** 2 / (2.0 * sigma)


def par_to_model(dataset, pars):
    """Update model in dataset with a list of free parameters factors"""
    for i, p in enumerate(dataset.models.parameters.free_parameters):
        p.factor = pars[i]


def ln_uniform_prior(dataset):
    """LogLike associated with prior and data/model evaluation.

    Return probability of parameter values according to prior knowledge.
    Parameter limits should be done here through uniform prior ditributions
    """
    logprob = 0
    for par in dataset.models.parameters.free_parameters:
        logprob += uniform_prior(par.value, par.min, par.max)

    return logprob


def lnprob(pars, dataset):
    """Estimate the likelihood of a model including prior on parameters."""
    # Update model parameters factors inplace
    for factor, par in zip(pars, dataset.models.parameters.free_parameters):
        par.factor = factor

    lnprob_priors = ln_uniform_prior(dataset)

    # dataset.likelihood returns Cash statistics values
    # emcee will maximisise the LogLikelihood so we need -dataset.likelihood
    total_lnprob = -dataset.stat_sum() + lnprob_priors

    return total_lnprob


def lnprob_with_regularization(pars, dataset, reg_depth=1):
    """Estimate the likelihood of a model including a Tikhonov regularization term and prior on parameters."""
    # Update model parameters factors inplace
    for factor, par in zip(pars, dataset.models.parameters.free_parameters):
        par.factor = factor

    #lnprob_priors = ln_uniform_prior(dataset) # remove priors

    # Include the Tikhonov regularization
    B_arr = get_npred_Etrue(dataset)
    reg_term = compute_reg(B_arr, reg_depth)

    # dataset.likelihood returns Cash statistics values
    # emcee will maximisise the LogLikelihood so we need -dataset.likelihood
    total_lnprob = -dataset.stat_sum() - reg_term

    return total_lnprob


    
###############################     run mcmc with Tikhonov regularization    #########################################

def run_mcmc_finding_optimal_reg(dataset, nwalkers, nrun, threads=1, get_trace=False, backend=None):
    """ Run the MCMC sampler with different values of regularization depth; find the optimal depth and return the corresponding sampler object.

        get_trace (optional) : get resulting tr_data_arr and  tr_npred_arr for the given depth_arr
    """
    depth_arr = np.logspace(np.log10(0.2), np.log10(2), 60)

    tr_data_arr = np.zeros(len(depth_arr))
    tr_npred_arr = np.zeros(len(depth_arr))

    # calculate corresponding traces for the given depth_arr
    for i, depth in enumerate(depth_arr):
        log.info(f"\n Trying MCMC with regularization depth = {depth}")
        
        sampler = run_mcmc_with_reg(dataset, nwalkers=10, nrun=8000, reg_depth=depth, backend=None)
        
        tr_data_arr[i], tr_npred_arr[i] = check_regularization(sampler, dataset, nburn=500)
        log.info(f"Current stat_sum = {dataset.stat_sum()}")

    # Find the optimal reg. depth
    # Better to find it manually with resulting tr_data_arr, tr_npred_arr
    ratio = tr_npred_arr / tr_data_arr
    best_idx = np.argmin( (ratio-1)**2 )
    best_depth = depth_arr[best_idx]
    
    log.info(f"\n The optimal regularization depth = {best_depth}")
    log.info(f"With tr_cov_data = {tr_data_arr[best_idx]},  tr_npred_data = {tr_npred_arr[best_idx]}")
    log.info(f"stat_sum = {dataset.stat_sum()}")
    
    # re-run mcmc 
    sampler = run_mcmc_with_reg(dataset, nwalkers, nrun, reg_depth = best_depth, backend=backend)
    log.info("Saving the sampler with the optimal regularization depth...")

    log.info("Finished!")
    if get_trace:
        return depth_arr, tr_data_arr, tr_npred_arr, sampler
    else:
        return sampler


def run_mcmc_with_reg(dataset, nwalkers=12, nrun=5000, threads=1, reg_depth=1, backend=None):
    """Run the MCMC sampler.

    Parameters
    ----------
    dataset : `~gammapy.modeling.Dataset`
        Dataset
    nwalkers : int
        Number of walkers
    nrun : int
        Number of steps each walker takes
    threads : (optional)
        Number of threads or processes to use

    Returns
    -------
    sampler : `emcee.EnsembleSampler`
        sampler object containing the trace of all walkers.
    """
    import emcee

    dataset.models.parameters.autoscale()  # Autoscale parameters

    # Initialize walkers in a ball of relative size 0.5% in all dimensions if the
    # parameters have been fit, or to 10% otherwise
    # Handle source position spread differently with a spread of 0.1°
    # TODO: the spread of 0.5% below is valid if a pre-fit of the model has been obtained.
    # currently the run_mcmc() doesn't know the status of previous fit.
    p0var = []
    pars = []
    spread = 0.5 / 100
    spread_pos = 0.1  # in degrees
    for par in dataset.models.parameters.free_parameters:
        pars.append(par.factor)
        if par.name in ["lon_0", "lat_0"]:
            p0var.append(spread_pos / par.scale)
        else:
            p0var.append(spread * par.factor)

    ndim = len(pars)
    p0 = emcee.utils.sample_ball(pars, p0var, nwalkers)

    labels = []
    for par in dataset.models.parameters.free_parameters:
        labels.append(par.name)
        #if (par.min is np.nan) and (par.max is np.nan):
        #    log.warning(
        #        f"Missing prior for parameter: {par.name}.\nMCMC will likely fail!"
        #    )

    log.info(f"Free parameters: {labels}")

    ############### changed ####################
    #sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[dataset], threads=threads)
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, lnprob_with_regularization, args=[dataset, reg_depth], threads=threads, backend=backend
    )

    log.info(f"Starting MCMC sampling: nwalkers={nwalkers}, nrun={nrun}")
    for idx, result in enumerate(sampler.sample(p0, iterations=nrun)):
        if idx % (nrun / 4) == 0:
            log.info("{:5.0%}".format(idx / nrun))
    log.info("100% => sampling completed")
        
    return sampler


#############################################  Coner plot  #############################################

def plot_trace(sampler, dataset):
    """
    Plot the trace of walkers for every steps

    Parameters
    ----------
    sampler : `emcee.EnsembleSampler`
        Sampler object containing the trace of all walkers
    dataset : `~gammapy.modeling.Dataset`
        Dataset
    """
    import matplotlib.pyplot as plt

    free_param = dataset.models.parameters.free_parameters
    labels = [par.name for par in free_param]
    scales = np.array([par.scale for par in free_param])
    
    fig, axes = plt.subplots(len(labels), sharex=True)

    for idx, ax in enumerate(axes):
        ax.plot(sampler.chain[:, :, idx].T * scales[idx], "-k", alpha=0.2)
        ax.set_ylabel(labels[idx])

    plt.xlabel("Nrun")
    plt.show()


def plot_corner(sampler, dataset, nburn=0):
    """Corner plot for each parameter explored by the walkers.

    Parameters
    ----------
    sampler : `emcee.EnsembleSampler`
        Sampler object containing the trace of all walkers
    dataset : `~gammapy.modeling.Dataset`
        Dataset
    nburn : int
        Number of runs to discard, because considered part of the burn-in phase
    """
    from corner import corner

    free_param = dataset.models.parameters.free_parameters
    labels = [par.name for par in free_param]
    scales = np.array([par.scale for par in free_param])

    samples = sampler.chain[:, nburn:, :].reshape((-1, len(labels)))
    scaled_samples = samples * scales

    corner(scaled_samples, labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles=True)



#############################################  Get spectral points #############################################

def get_true_ene_edges(dataset):
    """ get centers of the true energy bins for given dataset. """
    model_name = dataset.models.names[0]
    eval = dataset.evaluators[model_name]

    true_ene_edges = eval.geom.axes['energy_true'].edges
    return true_ene_edges


    
def get_true_ene_cntr(dataset):
    """ get centers of the true energy bins for given dataset. """
    model_name = dataset.models.names[0]
    eval = dataset.evaluators[model_name]

    true_ene_edges = eval.geom.axes['energy_true'].edges

    true_ene_cntr = []
    for emin,emax in list(zip(true_ene_edges[:-1], true_ene_edges[1:])):
        ecntr = np.sqrt(emin*emax)
        true_ene_cntr.append(ecntr)
    
    true_ene_cntr = u.Quantity(true_ene_cntr)
    return true_ene_cntr
    

    
def get_dnde(sampler, dataset, nburn):
    """ Specified for the stepped pwl model. Get dnde values at centers of true energy bins
    
    return: dnde_arr, dnde_err_minus_arr, dnde_err_plus_arr """
    
    median, err_minus, err_plus = set_mcmc_results(sampler, dataset, nburn, print=False)
    true_ene_cntr = get_true_ene_cntr(dataset)
    
    dnde_arr = np.zeros(len(true_ene_cntr)) * u.Unit("cm-2 s-1 TeV-1")
    dnde_err_plus_arr = np.zeros(len(true_ene_cntr)) * u.Unit("cm-2 s-1 TeV-1")
    dnde_err_minus_arr = np.zeros(len(true_ene_cntr)) * u.Unit("cm-2 s-1 TeV-1")

    # get the starting index (n_start) from the name of the 1st norm_i parameter
    free_param = dataset.models.parameters.free_parameters
    param_start = free_param[0].name
    n_start = int(param_start.split("_")[1])
    
    for i in range(len(free_param)):
        # Specified for the free normalization factors defined in the stepped pwl model
        factor = dataset.models.parameters['amplitude'] * ( true_ene_cntr[i+n_start]/ dataset.models.parameters['reference'])**(-2)
        
        dnde_arr[i+n_start] = median[i] * factor
        dnde_err_minus_arr[i+n_start] = err_minus[i] * factor
        dnde_err_plus_arr[i+n_start] = err_plus[i] * factor
        
    
    dnde_arr = u.Quantity(dnde_arr)
    dnde_err_minus_arr = u.Quantity(dnde_err_minus_arr)
    dnde_err_plus_arr = u.Quantity(dnde_err_plus_arr)
    
    return dnde_arr, dnde_err_minus_arr, dnde_err_plus_arr
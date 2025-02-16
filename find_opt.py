import numpy as np
import matplotlib.pyplot as plt
import logging

from astropy import units as u
import uproot
from astropy.coordinates import Angle, SkyCoord
from regions import PointSkyRegion

from gammapy.modeling.models import GaussianSpatialModel, FoVBackgroundModel, TemplateNPredModel
from gammapy.datasets import MapDataset
from gammapy.makers import MapDatasetMaker
from gammapy.data import Observation

from my_sampling_v3 import run_mcmc_finding_optimal_reg, PiecewiseConstSpectralModel

from IPython.display import display, Math
from gammapy.data import DataStore, ObservationFilter
from gammapy.maps import Map, WcsGeom, MapAxis, RegionGeom
from gammapy.datasets import Datasets, SpectrumDatasetOnOff, SpectrumDataset, FluxPointsDataset
from gammapy.makers import SpectrumDatasetMaker, ReflectedRegionsBackgroundMaker, SafeMaskMaker, WobbleRegionsFinder, PhaseBackgroundMaker
from gammapy.makers.utils import make_theta_squared_table
from gammapy.modeling import Fit, Parameter
from gammapy.modeling.models import PowerLawSpectralModel, LogParabolaSpectralModel, ExpCutoffPowerLawSpectralModel, PiecewiseNormSpectralModel
from gammapy.modeling.models import SkyModel, Models, ConstantSpatialModel, create_crab_spectral_model
from gammapy.estimators import FluxPointsEstimator
import emcee


# Geminga
target_position = SkyCoord(ra=98.475638, dec=17.770253, unit="deg", frame="icrs")
on_region = PointSkyRegion(target_position)

logging.basicConfig(level=logging.INFO, 
                    format='%(message)s',
                    handlers=[
                        logging.FileHandler("mcmc_results_new_model/Geminga/geminga_output.log"),
                        logging.StreamHandler()
                    ])

# set energy axis
ereco = np.geomspace(5*u.GeV,50*u.TeV,30+1)
etrue = np.geomspace(5*u.GeV,50*u.TeV,int(30/1.4 + 1))

energy_axis = MapAxis.from_energy_edges(ereco,name="energy",unit='GeV')
energy_axis_true = MapAxis.from_energy_edges(etrue,name='energy_true',unit='GeV')

logging.info("Loading Geminga P2 data...")
dataset00 = SpectrumDatasetOnOff.read("../Dataset_reduced/GemingaP2_dataset00_stacked.fits" )

# pwl fit
spectral_model = PowerLawSpectralModel(
    amplitude=1e-9 * u.Unit("cm-2 s-1 TeV-1"),
    index=5,
    reference=32.15 * u.GeV,
)
model = SkyModel(spectral_model=spectral_model, name="gemingaP2_pwl")

dataset00.models = [model]

fit_pwl = Fit()

result_pwl = fit_pwl.run(datasets=dataset00)
model_best_pwl = model.copy()

logging.info(result_pwl)
logging.info(display(result_pwl.models.to_parameters_table()))

# fit a stepped-pwl model
source_amp = model_best_pwl.copy().spectral_model.amplitude
source_amp.value = source_amp.value

spectral_model = PowerLawSpectralModel(index=2.0,amplitude=source_amp, reference=32.15*u.GeV)
spectral_model.index.frozen = True
spectral_model.amplitude.frozen = True
spectral_model.amplitude.min = 0

# Center points for all of the edges
true_ene_edges = energy_axis_true.edges
true_ene_cntr = []
for emin,emax in list(zip(true_ene_edges[:-1], true_ene_edges[1:])):
    ecntr = np.sqrt(emin*emax)
    true_ene_cntr.append(ecntr)

true_ene_cntr = u.Quantity(true_ene_cntr)
ntrue_ene_cntr = len(true_ene_cntr)
nflux_points = ntrue_ene_cntr

# define a piecewise spectral function
Piecewise_const = PiecewiseConstSpectralModel(energy_edges = true_ene_edges)

for idx in range(nflux_points):
    #Piecewise.parameters['norm_%i'%idx].min = 1e-3
    Piecewise_const.parameters['norm_%i'%idx].min = 1e-3
    Piecewise_const.parameters['norm_%i'%idx].max = 1e+2
    
    if true_ene_edges[:-1][idx] < 12*u.GeV :
        Piecewise_const.parameters['norm_%i'%idx].value = 0
        Piecewise_const.parameters['norm_%i'%idx].frozen = True
    if true_ene_edges[1:][idx] > 100*u.GeV :
        Piecewise_const.parameters['norm_%i'%idx].value = 0
        Piecewise_const.parameters['norm_%i'%idx].frozen = True
    
# Define the test source
test_source = SkyModel(
    spectral_model=spectral_model* Piecewise_const,
    name="test_source",
)

models = [test_source]
models = Models(models)
dataset00.models = models

logging.info("Starting stepped pwl minuit fitting...")
# try minuit for comparison
fit_stepped_pwl = Fit(backend='minuit')

result_stepped_pwl = fit_stepped_pwl.run(datasets=dataset00)

logging.info(result_stepped_pwl)
logging.info(test_source.spectral_model)


logging.info("Start running MCMC...")
nwalkers = 12
filename = "mcmc_results_new_model/Geminga/sampler_opt.h5"
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers=nwalkers, ndim=4)

depth_arr, tr_data_arr, tr_npred_arr, stat_arr, reg_arr, sampler_opt = run_mcmc_finding_optimal_reg(dataset00, nwalkers=nwalkers, nrun=8000, threads=1, get_trace=True, backend=backend)

np.save("mcmc_results_new_model/Geminga/geminga_depth_arr.npy", depth_arr)
np.save("mcmc_results_new_model/Geminga/geminga_tr_data_arr.npy", tr_data_arr)
np.save("mcmc_results_new_model/Geminga/geminga_tr_npred_arr.npy", tr_npred_arr)
np.save("mcmc_results_new_model/Geminga/geminga_stat_arr.npy", stat_arr)
np.save("mcmc_results_new_model/Geminga/geminga_reg_arr.npy", reg_arr)
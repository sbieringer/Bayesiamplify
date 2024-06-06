import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal

import sys
sys.path.append('./src/')
sys.path.append('./src/models/')

from cond_CFM import CNF, FlowMatchingLoss

sys.path.append('../MCMC_by_backprob/src/')
from MCMC_Adam import MCMC_by_bp

import numpy as np
import normflows as nf
import os
from tqdm import tqdm
from typing import *
from zuko.utils import odeint
from scipy.special import kl_div
from scipy.stats import wasserstein_distance

from models.custom_linear_flipout import custom_LinearFlipout as LinearFlipout, Flipout_Dropout, LinearDropout
from bayesian_torch.layers.dropout import Dropout
from bayesian_torch.layers.batchnorm import BatchNorm1dLayer

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pickle

from src.utils import *
from src.dataloader import *
from src.plot_util import *

def mkdir(save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    return save_dir

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#amplification over number bins

r_test = 4.0
n_samples = 100_000
c_factor = 100.
p = 0
data_dim = 2

data_path = './data/' 
add_str_save = ''

approximate_gaussian_inference = True
MCMC = approximate_gaussian_inference == True
save_every = 10

p = 0
n_nodes = 32
n_layers = 3

#donut_args = {'u_mean':0, 'u_sigma':1, 'r_mean': 4.0, 'r_sigma': 1, 'm_cut': 'inf', 'gamma_scale': 2.}
#sampler = multidim_sampler(data_dim, "donut_gamma", save_path = data_path, **donut_args)
donut_args = {'u_mean':0, 'u_sigma':1, 'r_mean': 4.0, 'r_sigma': 1, 'm_cut': 'inf', 'gamma_scale': 10}
sampler = multidim_sampler(data_dim, "donut", save_path = data_path, **donut_args)

save_dir = mkdir(f'./models/CFM_VIB_k{c_factor}_cond')
#save_dir = mkdir(mkdir(save_dir + f'/donut_{data_dim}d{add_str_save}_{donut_args["r_mean"]}rmean_{donut_args["gamma_scale"]}gamma/')+f'{n_samples}pts/')
save_dir = mkdir(mkdir(save_dir + f'/donut_normal_{data_dim}d{add_str_save}_{donut_args["r_mean"]}rmean_{donut_args["gamma_scale"]}gamma/')+f'{n_samples}pts/')
ep = 250000

n_bins_array = np.array([50]) #np.array([2,4,6,8,10,12,14,16,18,20,24,28,32,40,50])

z = torch.randn(5_000_000, data_dim).to(device)
c = r_test*torch.ones((len(z),1)).to(device)
n_stat_epis = 10 #if approximate_gaussian_inference else MCMC_samples

quant_list_list_path = f'./figs/quant_list_list_{donut_args["gamma_scale"]}gamma.pkl'
if not f'quant_list_list_{donut_args["gamma_scale"]}gamma.pkl' in os.listdir('./figs/'):
    data_true = cart_to_nsphere(sampler.sample_data(10_000_000)[0])
    quant_list_list = []
    for n_quantile in tqdm(n_bins_array):
        quant_list = quantiles(data_true, n_quantile, verbose = False)
        quant_list = (np.concatenate((np.asarray([-100]), np.asarray(quant_list[0]), np.asarray([100])), 0),
                    np.concatenate((np.asarray([0]), np.asarray(quant_list[1]), np.asarray([2*np.pi])), 0))
        quant_list_list.append(quant_list)
    del data_true
    with open(quant_list_list_path, 'wb') as f:
        pickle.dump(quant_list_list, f)
else:
    with open(quant_list_list_path, 'rb') as file: 
        quant_list_list = pickle.load(file) 

for n_mod in range(0,5):
    m_list_dir =  save_dir + f'/{n_mod}/' #+ '/AdamMCMC_models/' 
    #m_list_dir += f'0.05sigma_10.0temp_1e-05lr/'
    quantvals_nstatepis = []

    for n in tqdm(range(n_stat_epis)):

        if approximate_gaussian_inference:
            if p != 0:
                model = CNF(data_dim, conds = 1, hidden_features=[n_nodes] * n_layers, layer = Flipout_Dropout, layer_kwargs={'p': p})
            else:
                model = CNF(data_dim, conds = 1, hidden_features=[n_nodes] * n_layers, layer = LinearFlipout)

        else:
            if p != 0:
                model = CNF(data_dim, conds = 1, hidden_features=[n_nodes] * n_layers, layer = LinearDropout,  layer_kwargs={'p': p})
            else:
                model = CNF(data_dim, conds = 1, hidden_features=[n_nodes] * n_layers)
                
        model.to(device)
        model.load_state_dict(torch.load(m_list_dir + f"model_{ep}.pth"))

        if approximate_gaussian_inference:
            for layer in model.modules():
                if isinstance(layer, LinearFlipout):
                    layer._dnn_to_bnn_flag = True
                    layer.auto_sample = False 
                    layer.sample_weights()
        elif MCMC:
            model.load_state_dict(torch.load(m_list_dir + f"AdamMCMC_model_{n*save_every}.pth"))

        generated_data = model.decode(z, cond=c).detach().cpu().numpy()

        quantvals = []
        for i_nq, n_quantile in enumerate(n_bins_array):
            quantvals.append(quantile_values_via_hist(quant_list_list[i_nq], cart_to_nsphere(generated_data)))
        quantvals_nstatepis.append(quantvals)

        del generated_data

    quantvals_nstatepis_arr_list = [np.array([quantvals_nstatepis[i_stat][i_nq] for i_stat in range(len(quantvals_nstatepis))]) for i_nq in range(len(n_bins_array))]

    with open(m_list_dir + f'quantvals_nstatepis_arr_list.pkl', 'wb') as f:
        pickle.dump(quantvals_nstatepis_arr_list, f)
    with open(m_list_dir + f'n_bins_array.npy', 'wb') as f:
        np.save(f, n_bins_array)

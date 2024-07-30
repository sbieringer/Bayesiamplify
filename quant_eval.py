# %% [markdown]
# Evaluate the Models in Quantiles

# %%
import torch

import sys
sys.path.append('./src/')
sys.path.append('./src/models/')

from cond_CFM import CNF

import numpy as np
import os
from tqdm import tqdm

from models.custom_linear_flipout import custom_LinearFlipout as LinearFlipout, Flipout_Dropout, LinearDropout

import pickle
from src.utils import *
from src.dataloader import *
from src.plot_util import *
from argparse import ArgumentParser, ArgumentTypeError


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')

parser = ArgumentParser()
parser.add_argument('--n_samples', type=float, default=10_000)
parser.add_argument('--r_test', type=float, default=4.)
parser.add_argument('--data_dim', type=int, default=2)

parser.add_argument('--approximate_gaussian_inference', type=str2bool, default=False)
parser.add_argument('--linear', type=str2bool, default=False)
parser.add_argument('--long', type=str2bool, default=True)
parser.add_argument('--scaled', type=str2bool, default=False)
parser.add_argument('--c_factor', type=float, default=0)

parser.add_argument('--start', type=int, default=0)
parser.add_argument('--stop', type=int, default=5)
parser.add_argument('--n_stat', type=int, default=50)

runargs = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

############
### data ###
############

r_test = runargs.r_test
n_samples = runargs.n_samples
data_dim = runargs.data_dim
donut_shape = "donut_gamma"
add_str_save = ''

data_path = './data/' 

#donut_args = {'u_mean':0, 'u_sigma':1, 'r_mean': 4.0, 'r_sigma': 1, 'm_cut': 'inf', 'gamma_scale': 2.}
#sampler = multidim_sampler(data_dim, "donut_gamma", save_path = data_path, **donut_args)
donut_args = {'u_mean':0, 'u_sigma':1, 'r_mean': 4.0, 'r_sigma': 1, 'm_cut': 'inf', 'gamma_scale': 2.}
sampler = multidim_sampler(data_dim, donut_shape, save_path = data_path, **donut_args)

#################################################
### calculate quantiles or set histogram bins ###
#################################################

z = torch.randn(10_000_000, data_dim).to(device)
batch_size_sample = 20_000_000
c = r_test*torch.ones((len(z),1)).to(device)

linear = runargs.linear
quantile_bins = np.array([2,3,4,5,7,10,15,20,30,40,50,70,100,150,200,300,400,500,700]) if runargs.long else np.array([5,10,50]) 
n_bins_array = np.array([50]) if linear else quantile_bins 

if linear:
    name_add = '_linear'
elif runargs.long:
    name_add = '_long'
else:
    name_add = ''

if runargs.scaled:
    name_add += "_scaled"
    n_bins_array = np.array([2,3,4,5,7,10,15,20,30,40,50,70,100,200,500])
#if runargs.approximate_gaussian_inference==False:
    name_add += "_50draws"


quant_list_list_name = f'quant_list_list_{donut_args["gamma_scale"]}gamma{name_add}.pkl'
quant_list_list_path = f'./figs/' + quant_list_list_name

if not quant_list_list_name in os.listdir('./figs/') or linear:
    data_true = cart_to_nsphere(sampler.sample_data(10_000_000)[0])
    quant_list_list = []
    for n_quantile in tqdm(n_bins_array):
        if linear:
            quant_list = (np.linspace(4,6,n_quantile), np.linspace(0,2*np.pi,n_quantile))
        else:
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

assert len(quant_list_list) == len(n_bins_array), f"the loaded quant_list_list has {len(quant_list_list)} entries, but n_bins_array has {len(n_bins_array)}"

####################
### model params ###
####################

approximate_gaussian_inference = runargs.approximate_gaussian_inference
MCMC = approximate_gaussian_inference == False

p = 0
n_nodes = 32
n_layers = 3

assert runargs.c_factor != 0, 'Please set the noise value' if MCMC else 'Please set the prior KLD balance'
c_factor = runargs.c_factor
#c_factor = .0001
ep = 250000 if approximate_gaussian_inference else None
if approximate_gaussian_inference:
    if runargs.c_factor == 1.0:
        ep = 250000
    elif runargs.c_factor == 5.0:
        ep =200000
    elif runargs.c_factor == 10.0:
        ep = 180000
    elif runargs.c_factor == 50.:
        ep = 75000
    elif runargs.c_factor == 100.:
        ep = 40000

n_stat_epis = runargs.n_stat #if approximate_gaussian_inference else 10
save_every = 100

if approximate_gaussian_inference:
    save_dir = f'./models/CFM_VIB_k{c_factor}_cond'
else:
    save_dir = f'./models/CFM_cond'

save_dir = save_dir + f'/{donut_shape}_{data_dim}d{add_str_save}_{donut_args["r_mean"]}rmean_{donut_args["gamma_scale"]}gamma/'+f'{n_samples}pts/'
m_list_dir = save_dir

#################################
### calculate quantile values ###
#################################

n_reps = range(runargs.start,runargs.stop)
calc = True #False

if calc:
    for n_mod in n_reps:
        m_list_dir =  save_dir + f'/{n_mod}/' 
        if MCMC:
            m_list_dir += '/AdamMCMC_models/' 
            m_list_dir += f'{c_factor}sigma_1.0temp_0.001lr_50.0sigma_adam_dir_denom/'
        quantvals_nstatepis = []

        if approximate_gaussian_inference:
            if p != 0:
                model = CNF(data_dim, conds = 1, hidden_features=[n_nodes] * n_layers, layer = Flipout_Dropout, layer_kwargs={'p': p})
            else:
                model = CNF(data_dim, conds = 1, hidden_features=[n_nodes] * n_layers, layer = LinearFlipout)
            model.load_state_dict(torch.load(m_list_dir + f"model_{ep}.pth"))
        else:
            if p != 0:
                model = CNF(data_dim, conds = 1, hidden_features=[n_nodes] * n_layers, layer = LinearDropout,  layer_kwargs={'p': p})
            else:
                model = CNF(data_dim, conds = 1, hidden_features=[n_nodes] * n_layers)

        model.to(device)

        for n in tqdm(range(n_stat_epis)):
            if approximate_gaussian_inference:
                for layer in model.modules():
                    if isinstance(layer, LinearFlipout):
                        layer._dnn_to_bnn_flag = True
                        layer.auto_sample = False 
                        layer.sample_weights()
            elif MCMC:
                model.load_state_dict(torch.load(m_list_dir + f"AdamMCMC_model_{n*save_every}.pth"))

            if not runargs.scaled:
                with torch.no_grad():
                    if len(z) > batch_size_sample:
                        generated_data = np.zeros((len(z), data_dim)) 
                        for n_z in range(len(z)//batch_size_sample):
                            z_sample_temp = z[n_z*batch_size_sample:(n_z+1)*batch_size_sample]
                            c_sample_temp = c[n_z*batch_size_sample:(n_z+1)*batch_size_sample]
                            generated_data[n_z*batch_size_sample:(n_z+1)*batch_size_sample] = model.decode(z_sample_temp, cond=c_sample_temp).detach().cpu().numpy()
                    else:
                        generated_data =  model.decode(z, cond=c).detach().cpu().numpy()

            quantvals = []
            for i_nq, n_quantile in enumerate(n_bins_array):
                if runargs.scaled:
                    with torch.no_grad():
                        z = torch.randn(int(1e3*n_quantile**2), data_dim).to(device)
                        c = r_test*torch.ones((len(z),1)).to(device)

                        if len(z) > batch_size_sample:
                            generated_data = np.zeros((len(z), data_dim)) 
                            for n_z in range(len(z)//batch_size_sample):
                                z_sample_temp = z[n_z*batch_size_sample:(n_z+1)*batch_size_sample]
                                c_sample_temp = c[n_z*batch_size_sample:(n_z+1)*batch_size_sample]
                                generated_data[n_z*batch_size_sample:(n_z+1)*batch_size_sample] = model.decode(z_sample_temp, cond=c_sample_temp).detach().cpu().numpy()
                        else:
                            generated_data =  model.decode(z, cond=c).detach().cpu().numpy()
                quantvals.append(quantile_values_via_hist(quant_list_list[i_nq], cart_to_nsphere(generated_data)))
            quantvals_nstatepis.append(quantvals)

            del generated_data

        quantvals_nstatepis_arr_list = [np.array([quantvals_nstatepis[i_stat][i_nq] for i_stat in range(len(quantvals_nstatepis))]) for i_nq in range(len(n_bins_array))]

        with open(m_list_dir + f'quantvals_nstatepis_arr_list{name_add}.pkl', 'wb') as f:
            pickle.dump(quantvals_nstatepis_arr_list, f)
        with open(m_list_dir + f'n_bins_array{name_add}.npy', 'wb') as f:
            np.save(f, n_bins_array)
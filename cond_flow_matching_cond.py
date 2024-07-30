# %% [markdown]
# 'Flow Matching in 100 LOC'-code https://gist.github.com/francois-rozet/fd6a820e052157f8ac6e2aa39e16c1aa 
# augmented with Bayesian Methods
# for a simple 2D toy model

# %%
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal

import numpy as np
import os
from tqdm import tqdm
from typing import *
from scipy.special import kl_div
from scipy.stats import wasserstein_distance

from argparse import ArgumentParser

from bayesian_torch.models.dnn_to_bnn import get_kl_loss

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pickle

# %%
import sys
sys.path.append('./src/')
sys.path.append('./src/models/')

from MCMC_Adam import MCMC_by_bp

from cond_CFM import CNF, FlowMatchingLoss
from models.custom_linear_flipout import custom_LinearFlipout as LinearFlipout, Flipout_Dropout, LinearDropout

from src.utils import *
from src.dataloader import *
from src.plot_util import *

# %%
def mkdir(save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    return save_dir

def smooth(x, kernel_size=5):
    if kernel_size == 1:
        return x
    else:
        assert kernel_size % 2 != 0
        x_shape = x.shape
        x_tmp = np.array([x[i:x_shape[0]-kernel_size+i+1] for i in range(kernel_size)])
        edge1 = x[:int((kernel_size-1)/2)]
        edge2 = x[-int((kernel_size-1)/2):]
        x_out = np.concatenate((edge1, np.mean(x_tmp, 0),edge2),0)
        assert x_shape == x_out.shape
        return x_out #np.mean(np.array(x).reshape(-1, kernel_size),1)

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
parser.add_argument('--approximate_gaussian_inference', type=str2bool, default=False)
parser.add_argument('--n_points', type=int, default=1_000_000)
parser.add_argument('--n_rep', type=int, default=0)

parser.add_argument('--k', type=float, default=10)
parser.add_argument('--r_mean', type=float, default=4.)
parser.add_argument('--gamma_scale', type=float, default=2.)

parser.add_argument('--inv_temp', type=float, default=1.)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--noise', type=float, default=0.05)
parser.add_argument('--sigma_adam_dir_denom', type=float, default=50.)
runargs = parser.parse_args()
    
if runargs.r_mean == 0:
    runargs.r_mean = 'triag'
    
# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# %%
approximate_gaussian_inference = runargs.approximate_gaussian_inference
MCMC = approximate_gaussian_inference == False 

train = True
train_MCMC = MCMC 

assert MCMC != approximate_gaussian_inference; "choose either MCMC- or VI-Bayesian"

c_factor = runargs.k

if approximate_gaussian_inference:
    save_dir = mkdir(f'./models/CFM_VIB_k{c_factor}_cond')
else:
    save_dir = mkdir('./models/CFM_cond/')

# %%
#######################
### Define the data ###
#######################

from dataloader import *
from torch.utils.data import Dataset

class cond_dataset(Dataset):
    def __init__(self, x, c):
        super(Dataset, self).__init__()
        self.x = x
        self.c = c

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.c[index]

add_str_save = ''
data_dim = 2 #5
save_as_try = runargs.n_rep
n_samples = runargs.n_points
donut_shape = "donut_gamma"
donut_args = {'u_mean':0, 'u_sigma':1, 'r_mean': runargs.r_mean, 'r_sigma':1, 'm_cut': 'inf', 'gamma_scale': runargs.gamma_scale}

save_dir = mkdir(mkdir(mkdir(save_dir + f'/{donut_shape}_{data_dim}d{add_str_save}_{runargs.r_mean}rmean_{runargs.gamma_scale}gamma/')+f'{n_samples}pts/')+f'{runargs.n_rep}/')

data_path = './data/' 

#sampler = multidim_sampler(data_dim, "donut_gamma", save_path = data_path, **donut_args)
sampler = multidim_sampler(data_dim, donut_shape, save_path = data_path, **donut_args)

full_data = sampler.sample_data(n_samples, save_as_try = save_as_try)
test_data = sampler.sample_data(n_samples//10)
full_dataset = cond_dataset(full_data[0], full_data[1])
test_dataset = cond_dataset(test_data[0], test_data[1])

n_batches = 10
batch_size = n_samples//n_batches
dataloader = torch.utils.data.DataLoader(full_dataset, batch_size=batch_size, shuffle=True, num_workers=10, pin_memory=True, persistent_workers=True)
dataloader_test = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True, num_workers=10, pin_memory=True, persistent_workers=True)

bins=np.linspace(-7,7,50)
H,_,_ =  np.histogram2d(test_data[0][:,0], test_data[0][:,1], bins=bins)

# %%
########################
### Define the Model ###
########################
p = 0
n_nodes = 32
n_layers = 3

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

for layer in model.modules():
    if isinstance(layer, LinearFlipout):
        layer._dnn_to_bnn_flag = True
        layer.auto_sample = False 

print(f"initiated model with {sum(p.numel() for p in model.parameters())} parameters")

# %%
#####################
### Training Loop ###
#####################

lr = 1e-3
lr_decay = 1#0.9995
weight_decay = 0

epochs = 250001 if approximate_gaussian_inference else 2501

ep_start = 0 #0

if train:
    cfm_loss = FlowMatchingLoss(model)
    optim = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.99,0.99)) #torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, lr_decay)
    model.train()

    if ep_start == 0:
        loss_hist, loss_hist_KLD, loss_hist_test, loss_hist_test_ep, nlog_likl_hist, kl_div_hist, w1_hist = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]),  np.array([]),  np.array([])

    for ep in tqdm(range(ep_start, ep_start+epochs)):
        optim.zero_grad()
        
        for x,c in dataloader:
            optim.zero_grad()
            x,c = x.to(device), c.to(device)
            
            for layer in model.modules():
                if hasattr(layer, 'sample_weights'):
                    layer.sample_weights()

            # Compute loss
            loss_orig = cfm_loss(x, c)
            kl_loss = get_kl_loss(model)/len(x) if approximate_gaussian_inference else torch.zeros_like(loss_orig)
            loss = loss_orig + c_factor*kl_loss

            # Do backprop and optimizer step
            if ~(torch.isnan(loss) | torch.isinf(loss)):
                loss.backward()
                optim.step()
                scheduler.step()
            
        # Log loss
        loss_hist = np.append(loss_hist, loss_orig.to('cpu').data.numpy())
        loss_hist_KLD = np.append(loss_hist_KLD, kl_loss.to('cpu').data.numpy())

        if ep%100 == 0:
            loss_test, nlog_likl = torch.Tensor([0]).to('cpu'), torch.Tensor([0]).to('cpu')
            for x, c in dataloader_test:
                x,c = x.to(device), c.to(device)
                loss_test += cfm_loss(x, c).to('cpu')*len(x)
                if MCMC:
                    nlog_likl -= torch.sum(model.log_prob(x,c)).to('cpu')
                
            z = torch.randn(len(c), data_dim).to(device)
            generated_data = model.decode(z, cond=c).detach().cpu().numpy()
            H_gen,_,_ =  np.histogram2d(generated_data[:,0], generated_data[:,1], bins=bins)
            w1 = wasserstein_distance(H.flatten(), H_gen.flatten())
            kl_div_test = np.sum(np.nan_to_num(kl_div(H, H_gen), 0, 0, 0))

            loss_hist_test = np.append(loss_hist_test, loss_test.to('cpu').data.numpy()/len(dataloader_test.dataset))
            loss_hist_test_ep= np.append(loss_hist_test_ep, ep)
            nlog_likl_hist = np.append(nlog_likl_hist, nlog_likl.to('cpu').data.numpy()/len(dataloader_test.dataset))
            kl_div_hist = np.append(kl_div_hist, kl_div_test)
            w1_hist = np.append(w1_hist, w1)

        if ep%250 == 0:
            torch.save(model.state_dict(), save_dir + f"model_{ep}.pth")

    torch.save(model.state_dict(), save_dir + f"model_{ep}.pth")

    with open(save_dir + f'loss_hist_{ep}.npy', 'wb') as f:
        np.save(f, loss_hist)
    with open(save_dir + f'loss_hist_KLD_{ep}.npy', 'wb') as f:
        np.save(f, loss_hist_KLD)
    with open(save_dir + f'loss_hist_test_{ep}.npy', 'wb') as f:
        np.save(f, loss_hist_test)
    with open(save_dir + f'loss_hist_test_ep_{ep}.npy', 'wb') as f:
        np.save(f, loss_hist_test_ep)
    with open(save_dir + f'nlog_likl_hist_{ep}.npy', 'wb') as f:
        np.save(f, nlog_likl_hist)
    with open(save_dir + f'kl_div_hist_{ep}.npy', 'wb') as f:
        np.save(f, kl_div_hist)
    with open(save_dir + f'w1_hist_{ep}.npy', 'wb') as f:
        np.save(f, w1_hist)

    model.eval()

else:
    ep = epochs-1

    with open(save_dir + f'loss_hist_{ep}.npy', 'rb') as f:
        loss_hist = np.load(f)
    with open(save_dir + f'loss_hist_KLD_{ep}.npy', 'rb') as f:
        loss_hist_KLD =  np.load(f)
    with open(save_dir + f'loss_hist_test_{ep}.npy', 'rb') as f:
        loss_hist_test =  np.load(f)
    with open(save_dir + f'loss_hist_test_ep_{ep}.npy', 'rb') as f:
        loss_hist_test_ep =  np.load(f)
    with open(save_dir + f'nlog_likl_hist_{ep}.npy', 'rb') as f:
        nlog_likl_hist =  np.load(f)
    with open(save_dir + f'kl_div_hist_{ep}.npy', 'rb') as f:
        kl_div_hist =  np.load(f)
    with open(save_dir + f'w1_hist_{ep}.npy', 'rb') as f:
        w1_hist =  np.load(f)

    model.load_state_dict(torch.load(save_dir + f"model_{ep}.pth"))
    print('loaded model from ' + save_dir + f"model_{ep}.pth")

    model.eval()

# %%
#####################
### AdamMCMC Loop ###
#####################

lr = runargs.lr
lr_decay = 1 #0.9995

temp = runargs.inv_temp #10
sigma = runargs.noise #05
sigma_adam_dir_denom = runargs.sigma_adam_dir_denom

if MCMC and train_MCMC:

    save_every = 100
    MCMC_samples = 50 #10
    m_list_dir =  mkdir(save_dir + '/AdamMCMC_models/')

    MCMC_epochs = save_every*MCMC_samples+1

    m_list_dir= mkdir(m_list_dir + f'{sigma}sigma_{temp}temp_{lr}lr_{sigma_adam_dir_denom}sigma_adam_dir_denom/')

    loop_kwargs = {
                'MH': True, #this is a little more than x2 runtime
                'verbose': MCMC_epochs<10,
                'fixed_batches': True,
                'sigma_adam_dir': sum(p.numel() for p in model.parameters())/sigma_adam_dir_denom if sigma_adam_dir_denom!=0 else 0,
                'extended_doc_dict': False,
                'full_loss': None,
    }

    optim = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.99,0.99))
    model.device = device

    AdamMCMC = MCMC_by_bp(model, optim, temp, sigma)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, lr_decay)
    model.train()

    acc_hist, b_hist = np.array([]), np.array([])
    loss_hist, loss_hist_test, loss_hist_test_ep, kl_div_hist, w1_hist = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    maxed_out_mbb_batches = 0
    first = 0
    for ep in tqdm(range(MCMC_epochs)):        
        for x,c in dataloader:
            optim.zero_grad()
            x,c = x.to(device), c.to(device)
            
            # Compute log Likl
            loss = lambda: -torch.sum(model.log_prob(x, c))*n_samples/len(x)

            if first < 5:
                l = loss()
                l.backward()
                optim.step()
                scheduler.step()
                first += 1

            else:
                # Do backprop and optimizer step
                #t1= time()
                maxed_out_mbb_batches += 1
                _,a,b,sigma,stop_dict = AdamMCMC.step(loss, **loop_kwargs)
                #print(f't_update: {time()-t1:4.4} s')
                
                if b: 
                    maxed_out_mbb_batches  = 0
                if maxed_out_mbb_batches > 100:
                    print('MBB sampling is not convergent, reinitializing the chain')
                    AdamMCMC.start = True #This is a hot fix to not get the optimizer stuck to often

                scheduler.step()
                acc_hist = np.append(acc_hist, a.to('cpu').data.numpy())
                b_hist = np.append(b_hist, b)

        if ep%100 == 0:
            loss_test, nlog_likl = 0, 0
            for x,c in dataloader_test:
                x,c = x.to(device), c.to(device)
                loss_test -= torch.sum(model.log_prob(x, c))
                
            z = torch.randn(len(c), data_dim).to(device)
            generated_data = model.decode(z, cond=c).detach().cpu().numpy()
            H_gen,_,_ =  np.histogram2d(generated_data[:,0], generated_data[:,1], bins=bins)
            w1 = wasserstein_distance(H.flatten(), H_gen.flatten())
            kl_div_test = np.sum(np.nan_to_num(kl_div(H, H_gen), 0, 0, 0))

            loss_hist_test = np.append(loss_hist_test, loss_test.to('cpu').data.numpy()/len(dataloader_test.dataset))
            loss_hist_test_ep= np.append(loss_hist_test_ep, ep)
            kl_div_hist = np.append(kl_div_hist, kl_div_test)
            w1_hist = np.append(w1_hist, w1)
            
        # Log loss
        loss_hist = np.append(loss_hist, loss().to('cpu').data.numpy())

        if ep%save_every==0:
            torch.save(model.state_dict(), m_list_dir + f"AdamMCMC_model_{ep}.pth")

    with open(m_list_dir + f'AdamMCMC_losses_{ep}.npy', 'wb') as f:
        np.save(f, loss_hist)
    with open(m_list_dir + f'AdamMCMC_acc_{ep}.npy', 'wb') as f:
        np.save(f, acc_hist)
    with open(m_list_dir + f'loss_hist_{ep}.npy', 'wb') as f:
        np.save(f, loss_hist)
    with open(m_list_dir + f'loss_hist_test_{ep}.npy', 'wb') as f:
        np.save(f, loss_hist_test)
    with open(m_list_dir + f'loss_hist_test_ep_{ep}.npy', 'wb') as f:
        np.save(f, loss_hist_test_ep)
    with open(m_list_dir + f'kl_div_hist_{ep}.npy', 'wb') as f:
        np.save(f, kl_div_hist)
    with open(m_list_dir + f'w1_hist_{ep}.npy', 'wb') as f:
        np.save(f, w1_hist)

import torch
from copy import deepcopy as dc
from numpy import pi, sqrt, log
from numpy.random import rand
from time import time, sleep

def diff(params1, params2):
    diff = params1[0] - params2[0]
    return diff.detach()

def log_normal(diff_eval, len2_diff_eval, sigma, diff_adam=None, len2_diff_adam=0, sigma_adam_dir=0, verbose = False):
    with torch.no_grad():
        d = len(diff_eval)
        
        difference_prod = torch.inner(diff_eval, diff_adam) if diff_adam is not None else 0
        directional_factor = 1/(sigma**4/sigma_adam_dir**2 + sigma**2*len2_diff_adam) if diff_adam is not None else 0
        log_norm = -d*log(sigma) if diff_adam is None else -d*log(sigma)-torch.log((sigma_adam_dir/sigma)**2*len2_diff_adam+1)  #should include a 2\pi**d/2 but that cancels anyhow 
        
        log_normal_out = log_norm-0.5*(len2_diff_eval/sigma**2 - directional_factor*difference_prod**2)
        return log_normal_out

def accept_prop(temp, old_loss, prop_loss, old_state, new_state, prop_state, next_state, sigma, sigma_next, sigma_adam_dir = 0, sigma_adam_dir_next = 0, verbose = False):
    with torch.no_grad():
        diff_old = diff(old_state, next_state)
        diff_old_adam = diff(old_state, new_state) if sigma_adam_dir is not None else None
        diff_prop = diff(prop_state, new_state)
        diff_prop_adam = diff(prop_state, next_state) if sigma_adam_dir is not None else None
        
        len2_diff_old, len2_diff_prop = torch.sum(diff_old**2), torch.sum(diff_prop**2)
        len2_diff_old_adam = torch.sum(diff_old_adam**2)if sigma_adam_dir is not None else None  
        len2_diff_prop_adam = torch.sum(diff_prop_adam**2) if sigma_adam_dir is not None else None 
        
        prop_prob_old = log_normal(diff_old, len2_diff_old, sigma_next, diff_prop_adam, len2_diff_old_adam, sigma_adam_dir_next, verbose) 
        prop_prob_prop = log_normal(diff_prop, len2_diff_prop, sigma, diff_old_adam, len2_diff_prop_adam, sigma_adam_dir, verbose)
        
        val = torch.exp(temp*(old_loss-prop_loss) + prop_prob_old - prop_prob_prop)# + log_norms)

        val_ret = torch.nan_to_num(val, 0, posinf = 1, neginf = 0)
        val_ret = torch.clip(val_ret, None, 1e6)
    
    if verbose:
        print(f'acc_prob: {val_ret} \nloss old: {old_loss}, prop loss: {prop_loss} \nacceptance on loss: {torch.exp(temp*(old_loss-prop_loss))} \nacceptance on proposal distribution {torch.exp(prop_prob_old - prop_prob_prop)}')

    return val_ret, val, prop_prob_old-prop_prob_prop, 0.5/sigma**2*len2_diff_prop-0.5/sigma_next**2*len2_diff_old

class MCMC_by_bp():
    def __init__(self, model, optimizer, temperature, sigma=None, sigma_adam_dir = None, **kwargs):
        '''
        class implementing the cycSGMH algorithm. 
        Can be used similar to torchch.optimizer.Optimizer instances, but loss needs to be defined for fixed newtork weights.

        args: model - torch.nn.Module: NN model to optimize with optimizer, needed to give acces to model.stat_dict() 
              optimizer - torch.optimizer.Optimizer: instance of the torch.optimizer class already initialized with the parameters in question and the hyperparameters of choice
              temperature - FLOAT: temperature parameter. Higher parameter makes the Gibss-posterior sharper and the uncertainty smaller.
              sigma - 'dynamic', None, FLOAT: if float standard deviation of the noise added to the update step (normailized by 1/\sqrt(n_weights))
                                              if 'dynamic' the size of the update step is used
                                              if None optimzer.defaults.lr is used, if float sigma is used
                                              default = None
              sigma_adam_dir - FLOAT: if float standard deviation of the noise added to the update step in step direction (normailized by 1/\sqrt(n_weights)), default = None
                                                
        '''
        self.model = model
        self.opt = optimizer
        self.temp = temperature
        self.sigma = sigma
        self.sigma_adam_dir = sigma_adam_dir
        self.n_weights = torch.sum(torch.Tensor([x.numel() for x in self.model.parameters()])).item()
        
        self.old_state = None
        self.new_state = None

        self.old_loss = None
        self.old_loss_acc = None
        self.old_new_sqe = None
        self.start = True
                
    def step(self, loss, **kwargs):
        '''
        single step in the cycSGMH algorithm
        
        args: loss - callable: Returns the value of the loss for the current network weights (!!!)
            
        kwargs: verbose - BOOL: verbose output, default = False
                extended_doc_dict - BOOL: whether to return all important values of the step in a dict, default = False

                MH - BOOL: whethter to do a MH step, defalut = True
                full_loss - False, Callable: if callable full_loss() is called to determine the loss in MH step, if False loss() is called, default = False
                fixed_batches - BOOL: if False, the update from the MH correction is reused in the next step, default = True
                
                gamma_sigma_decay - FLOAT: exponential decay of sigma, default = None
                sigma_min - FLOAT: mimimim of exponential decay, default = 0
            
                sigma_factor - FLOAT: factor multiplied with sigma to get the proposal distribution standard deviation, important with self.sigma = 'dynamic', default = 1
                tau - FLOAT: exponent for self.sigma = 'dynamic', default = 1
                
                sigma_adam_dir - FLOAT: default = None            
        '''
        
        #kwargs
        fixed_batches = kwargs.get('fixed_batches', True) #fixed batches needs to be set (in theory) when MCMC-Adam is used with batched data (i.e. not Bernoulli-sampled batches) 
        verbose = kwargs.get('verbose', False)
        full_loss = kwargs.get('full_loss', False)
        use_full_loss = hasattr(full_loss, '__call__')

        extended_doc_dict = kwargs.get('extended_doc_dict', False)
        
        gamma_sigma_decay = kwargs.get('gamma_sigma_decay')
        sigma_min = kwargs.get('sigma_min', 0)
        tau = kwargs.get('tau', 1)
        sigma_factor = kwargs.get('sigma_factor', 1)
        sigma_adam_dir = kwargs.get('sigma_adam_dir') if kwargs.get('sigma_adam_dir') is not None else self.sigma_adam_dir
        
        MH = kwargs.get('MH', True)

        #initialize the doc dict
        if extended_doc_dict:
            with torch.no_grad():
                doc_dict = {'old_new_sqe': torch.Tensor([torch.nan]).to(self.model.device), 
                            'prop_next_sqe': torch.Tensor([torch.nan]).to(self.model.device), 
                            'loss_diff': torch.Tensor([torch.nan]).to(self.model.device), 
                            'old_loss': torch.Tensor([torch.nan]).to(self.model.device), 
                            'prop_loss': torch.Tensor([torch.nan]).to(self.model.device), 
                            'prob_diff': torch.Tensor([torch.nan]).to(self.model.device), 
                            'full_train_loss': torch.Tensor([torch.nan]).to(self.model.device), 
                        }
        else:
            doc_dict = {}
            
        #set old state
        self.old_state = self.get_params() #if this is not loaded at the beginning of each step, changes to e.g. the learning rate will be forgotten
        
        #calculate new state if the previous step can not be reused
        if self.start or fixed_batches:
            self.start = False
            
            self.old_loss = loss() #do the gradient descent step
            self.old_loss.backward()
            self.old_loss_acc = full_loss(self.model) if use_full_loss else self.old_loss
            self.opt.step()
            self.new_state = self.get_params()

            if self.sigma == 'dynamic':
                with torch.no_grad():
                    self.old_new_sqe = torch.sqrt(torch.sum(diff(self.old_state, self.new_state)**2))

                    
        if extended_doc_dict:
            doc_dict['old_new_sqe'] = self.old_new_sqe
            if use_full_loss:
                doc_dict['full_train_loss'] = self.old_loss_acc

                
        # calculate normailized sigmas 
        if self.sigma == 'dynamic':
            sigma = self.old_new_sqe.item()**tau
        elif self.sigma is not None:
            if gamma_sigma_decay is not None and self.sigma > sigma_min:
                self.sigma = self.sigma*gamma_sigma_decay
            sigma = dc(self.sigma)
        else:
            try: 
                sigma = self.print_lr()
            except:
                raise Exception('Please either specify a sigma or use a optimizer with a .lr attribute')
                
        sigma_norm_factor = sqrt(1/(self.n_weights)) 
        sigma *= sigma_norm_factor*sigma_factor
        sigma_adam_dir = sigma_adam_dir*sigma_norm_factor if sigma_adam_dir is not None else sigma_adam_dir
        
        if verbose:
            if self.sigma == 'dynamic':
                print(f'self.sigma {self.sigma}, sigma_norm_factor {sigma_norm_factor}, sigma_adam_dir {sigma_adam_dir} sigma for noise sampling of tau {sigma}, factor {sigma_factor}')

        # calculate proposed step
        if sigma_adam_dir is not None:
            sigma_item = sigma
            sampled_sigma = torch.normal(0, sigma_item, self.new_state[0].shape, device = self.model.device)
            sampled_sigma_adam = torch.normal(0, sigma_adam_dir, self.new_state[0].shape, device = self.model.device)
            prop_state = (sampled_sigma + self.new_state[0] + (self.new_state[0]-self.old_state[0])*sampled_sigma_adam, self.new_state[1])
        else:
            prop_state = (torch.normal(0, sigma_item, self.new_state[0].shape, device = self.model.device) + self.new_state[0], self.new_state[1])

        #do the MH correction
        if MH:
            self.set_params((prop_state[0], self.old_state[1])) #use same momentum for backward direction as for forwards!!!
            prop_state = (dc(prop_state[0]), prop_state[1]) #prevent magic value overwriting
            self.opt.zero_grad()

            prop_loss = loss()
            prop_loss.backward()
            prop_loss_acc = full_loss(self.model) if use_full_loss else prop_loss

            self.opt.step()
            next_state = self.get_params()

            with torch.no_grad():
                prop_next_sqe = None if self.sigma != 'dynamic' else torch.sqrt(torch.sum(diff(prop_state, next_state)**2)) 
                sigma_next = sigma if self.sigma != 'dynamic' else prop_next_sqe**tau*sigma_norm_factor*sigma_factor

            a, a_full,prob_diff_doc ,_ = accept_prop(self.temp, self.old_loss_acc, prop_loss_acc, self.old_state, self.new_state, 
                                                                                  prop_state, next_state, sigma, sigma_next, sigma_adam_dir = sigma_adam_dir, 
                                                                                  sigma_adam_dir_next = sigma_adam_dir, verbose = verbose) 
            if extended_doc_dict:
                doc_dict['prop_next_sqe'] = prop_next_sqe
                doc_dict['prob_diff'] = prob_diff_doc
                doc_dict['loss_diff'] = self.temp*(self.old_loss_acc-prop_loss_acc)
                doc_dict['old_loss'] = self.old_loss_acc
                doc_dict['prop_loss'] = prop_loss_acc

            #accept or reject the proposed state
            u = rand(1)
            if verbose:
                print(f"u {u} -> accepted {u <= a.cpu().numpy()}")
            if u <= a.cpu().numpy():
                self.old_state = prop_state 

                if not fixed_batches:
                    self.new_state = next_state 
                    self.old_new_sqe = prop_next_sqe
                    self.old_loss_acc = prop_loss_acc

                accepted = True
            else:
                accepted = False

        # or dont
        else:
            self.old_state = prop_state 
            a, a_full, accepted = torch.Tensor([1]), torch.Tensor([1]), True

        self.set_params(self.old_state)
        self.opt.zero_grad()
            
        return self.old_loss, a_full, accepted, sigma, doc_dict

    def get_params(self):
        return torch.nn.utils.parameters_to_vector(self.model.parameters()).detach(), dc(self.opt.state_dict())

    def set_params(self, out):
        torch.nn.utils.vector_to_parameters(out[0], self.model.parameters())
        self.opt.load_state_dict(out[1])
            
    def print_lr(self):
        return self.opt.param_groups[0]['lr']
        
    def zero_grad(self):
        self.opt.zero_grad()
        
    def state_dict(self):
        return self.opt.state_dict()
    
    def load_state_dict(self, state_dict):
        self.opt.load_state_dict(state_dict)
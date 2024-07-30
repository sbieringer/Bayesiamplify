
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform
from bayesian_torch.layers.base_variational_layer import BaseVariationalLayer_
from bayesian_torch.layers.flipout_layers.linear_flipout import LinearFlipout
from torch.quantization.qconfig import QConfig


class custom_LinearFlipout(LinearFlipout):
    def __init__(self,
                 in_features,
                 out_features,
                 prior_mean=0,
                 prior_variance=1,
                 posterior_mu_init=0,
                 posterior_rho_init=-3.0,
                 bias=True):
        """
        inits custom Linear Flipout layer to differentiate the weight sampling (weights need to be constant each call for ODE solving in continuous flow)
        """
        super().__init__(in_features,
                 out_features,
                 prior_mean,
                 prior_variance,
                 posterior_mu_init,
                 posterior_rho_init,
                 bias)
        
        self.auto_sample = True
        self.sample_weights()
        
    def sample_weights(self):
        sigma_weight = torch.log1p(torch.exp(self.rho_weight))
        eps_weight = self.eps_weight.data.normal_()    
        self.delta_weight = sigma_weight * eps_weight

        sigma_bias = torch.log1p(torch.exp(self.rho_bias))
        self.bias = (sigma_bias * self.eps_bias.data.normal_())

    def forward(self, x, return_kl=True, sample_weights=True):
        if self.dnn_to_bnn_flag:
            return_kl = False
        sigma_weight = torch.log1p(torch.exp(self.rho_weight))

        if self.auto_sample:
            self.sample_weights()

        # get kl divergence
        if return_kl:
            kl = self.kl_div(self.mu_weight, sigma_weight, self.prior_weight_mu,
                             self.prior_weight_sigma)

        bias = None
        if self.mu_bias is not None:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            if return_kl:
                kl = kl + self.kl_div(self.mu_bias, sigma_bias, self.prior_bias_mu,
                                      self.prior_bias_sigma)

        # linear outputs
        outputs = F.linear(x, self.mu_weight, self.mu_bias)
        if self.auto_sample:
            sign_input = x.clone().uniform_(-1, 1).sign()
            sign_output = outputs.clone().uniform_(-1, 1).sign()
        else: 
            sign_input = 1
            sign_output = 1
        x_tmp = x * sign_input
        perturbed_outputs_tmp = F.linear(x_tmp, self.delta_weight, self.bias)
        perturbed_outputs = perturbed_outputs_tmp * sign_output
        out = outputs + perturbed_outputs

        if self.quant_prepare:
            # quint8 quantstub
            x = self.quint_quant[0](x) # input
            outputs = self.quint_quant[1](outputs) # output
            sign_input = self.quint_quant[2](sign_input)
            sign_output = self.quint_quant[3](sign_output)
            x_tmp = self.quint_quant[4](x_tmp)
            perturbed_outputs_tmp = self.quint_quant[5](perturbed_outputs_tmp) # output
            perturbed_outputs = self.quint_quant[6](perturbed_outputs) # output
            out = self.quint_quant[7](out) # output

            # qint8 quantstub
            sigma_weight = self.qint_quant[0](sigma_weight) # weight
            mu_weight = self.qint_quant[1](self.mu_weight) # weight
            eps_weight = self.qint_quant[2](eps_weight) # random variable
            delta_weight =self.qint_quant[3](delta_weight) # multiply activation
            

        # returning outputs + perturbations
        if return_kl:
            return out, kl
        return out
    

class Flipout_Dropout(nn.Module):
    def __init__(self,
            in_features,
            out_features,
            prior_mean=0,
            prior_variance=1,
            posterior_mu_init=0,
            posterior_rho_init=-3.0,
            bias=True,
            p = 0):
        
        super().__init__()
        self.lf = LinearFlipout(in_features,
                 out_features,
                 prior_mean,
                 prior_variance,
                 posterior_mu_init,
                 posterior_rho_init,
                 bias)
        
        self.p = p
        self.out_features = out_features
        self.sample_weights()

    def forward(self, x):
        out = self.lf(x)
        if self.training:
            out *= self.dropout_mask
        return out
    
    def sample_weights(self):
        self.dropout_mask = (torch.rand([self.out_features]) >= self.p).reshape(1,-1).to(self.lf.mu_weight.device).type_as(self.lf.mu_weight)
        self.lf.sample_weights()

class LinearDropout(nn.Module):
    def __init__(self, input, output, p):
        super().__init__()
        self.lf = nn.Linear(input, output)

        self.p = p
        self.out_features = output
        self.sample_weights()

    def forward(self, x):
        out = self.lf(x)
        if self.training:
            out *= self.dropout_mask
        return out
    
    def sample_weights(self):
        self.dropout_mask = (torch.rand([self.out_features]) >= self.p).reshape(1,-1).to(self.lf.weight.device).type_as(self.lf.weight)
       
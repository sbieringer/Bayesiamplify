# define dataloading routines here

import numpy as np 
import os

def acot(x):
    out = np.arctan(1/x)
    out = out + (x<0)*np.pi
    return out
    
    
def cart_to_nsphere(in_cat):
    '''
    This functions maps input array in n-dim cartesian coodinates (x_1,...x_n) 
    to output array in n-dim spherical coordinates (r, \phi_1, ... \phi_{n-1})
    '''
    n_dim = in_cat.shape[1]
    if n_dim>1:
        r_out = np.expand_dims(np.sqrt(np.sum(in_cat**2, 1)),1)
        angle_out = []

        for d in range(n_dim-2):
            angle_out.append(acot(in_cat[:,d]/np.sqrt(np.sum(in_cat[:,d+1:]**2, 1))))

        angle_out.append(2*acot((in_cat[:,n_dim-2] + 
                                 np.sqrt(in_cat[:,n_dim-1]**2 + 
                                         in_cat[:,n_dim-2]**2 ))/in_cat[:,n_dim-1]))    
        angle_out = np.transpose(np.array(angle_out))
        out = np.concatenate((r_out, angle_out),1)
        return out
    else:
        return in_cat


class samplers():
    def __init__(self, save_path):
        self.save_path = save_path
    
    def load_data(self, load_try_nr, samp = None):
        file_list = os.listdir(self.save_path)

        if hasattr(self, 'save_name'):
            if samp is not None:
                try:
                    stringdex = self.save_name.index('samp') + 4
                    #stringdex_end = -1 #self.save_path.find()
                    load_name = self.save_name[:stringdex] + f'{samp}_'

                except:
                    load_name = self.save_name + f'_samp{samp}_'
            else:
                load_name = self.save_name
        
            file_list = [file for file in file_list if load_name in file]
        else:
            file_list = [file for file in file_list if file.startswith('fulldata')]

        try:
            single_file = [file for file in file_list if int(file[file.find('try')+3:file.index('.npy')])==load_try_nr]
        except:
            raise Exception(f'sampler.load_data: could not find dataset, load_try_nr {load_try_nr}')

        if len(single_file) > 1:
            raise Exception(f'sampler.load_data: found multiple matching datasets, set sampler.save_name, load_try_nr {load_try_nr}')
        else:                 
            return np.load(self.save_path + single_file[0])
    

class multidim_sampler(samplers):
    
    def __init__(self, n_dim, data_type="donut", save_path=None, **distr_kargs):
        self.type=data_type
        self.n_dim=n_dim
        self.distr_args=distr_kargs
        
        self.save_path = save_path 
        self.save_name = f'fulldata_{data_type}_dim{self.n_dim}'
        for key in distr_kargs:
            self.save_name += f'_{key}{distr_kargs[key]}'
        super().__init__(self.save_path)
        
    def sample_data(self, n, save_as_try=None, start=True):
        if start:
            #for recursive rejecition sampling
            self.x_counter = 0 
            self.n_sample = n
            
        if "donut" in self.type:
            u_mean=self.distr_args.get('u_mean', 0)
            u_sigma=self.distr_args.get('u_sigma', 1)

            r_mean_temp = self.distr_args.get('r_mean', 4)
            if r_mean_temp == 'exp':
                r_mean=4-np.clip(np.random.exponential(1/2, (n,1)),0, 4)
            if r_mean_temp == 'triag':
                r_mean=np.random.triangular(1,6,6, (n,1))
            else:
                r_mean=r_mean_temp*np.ones((n,1))
            r_sigma=self.distr_args.get('r_sigma', 1)
    
            u = np.random.normal(loc=(u_mean), scale=(u_sigma), size=(n,self.n_dim))
            norm=np.sum(u**2, 1, keepdims=True) **(0.5)
            gamma_scale = self.distr_args.get('gamma_scale', 'inf')
            if "gamma" in self.type and gamma_scale != 'inf':
                r = np.random.gamma(gamma_scale,1/gamma_scale, size=(n,1))
            else:
                r = np.random.normal(loc=0, scale=(r_sigma), size=(n,1))
                
            if "outer" in self.type:
                m_cut=self.distr_args.get('m_cut', 'inf')
                if m_cut == 'inf':
                    r = np.abs(r)
                else:
                    r_prob = 1/(1+np.exp(-m_cut*r))
                    u_draw = np.random.uniform(size=n) 
                    del_idx = np.argwhere(u_draw>r_prob.squeeze())
            r += r_mean
            
            if "outer" in self.type and m_cut != 'inf':
                x = np.delete(r*u/norm, del_idx, axis = 0)
                r_mean = np.delete(r_mean, del_idx, axis = 0)
                self.x_counter += len(x)
                if self.x_counter < self.n_sample:
                    data_tmp = self.sample_data(2*(self.n_sample-self.x_counter), None, False)
                    x = np.append(x, data_tmp[0], axis=0)
                    r_mean = np.append(r_mean, data_tmp[1], axis=0)
                    del data_tmp
                x = x[:self.n_sample]
                r_mean = r_mean[:self.n_sample]                
            else:
                x= r*u/norm
                        
        else:
            raise Exception("data_sampler: unknown sample distribution")
            
        #save datapoints if save_as_try   
        if save_as_try is not None:
            try:
                stringdex = self.save_name.index('samp') + 4
                #stringdex_end = -1 #self.save_path.find()
            
                self.save_name = self.save_name[:stringdex] + f'{n}'
                
            except:
                self.save_name = self.save_name + f'_samp{n}'
                
            np.savez(self.save_path + self.save_name + f'_try{save_as_try}', r_mean=r_mean, x=x)

        return x.astype(np.float32), r_mean.astype(np.float32)

    
class LaSeR_sampler(samplers):
    '''
    generate the 2d samples of the LaSeR paper : https://arxiv.org/pdf/2106.00792.pdf
    '''
    
    def __init__(self, save_path = None):
        
        self.n_dim=2
        self.save_path = save_path 
        self.save_name = 'fulldata_dim{0:d}'.format(self.n_dim)        
        super().__init__(self.save_path)
        
    def sample_eight(self, n, save_as_try=None):

        t = []

        n = n//2
        
        r1 = np.random.normal(4,1, [n,1])
        r2 = np.random.normal(4,1, [n,1])
        phi1 = np.random.uniform(0,2*np.pi, [n,1])
        phi2 = np.random.uniform(0,2*np.pi, [n,1])

        x1 = r1 * np.cos(phi1) + 4
        x2 = r2 * np.cos(phi2) - 4
        y1 = r1 * np.sin(phi1)
        y2 = r2 * np.sin(phi2)

        t1 = np.concatenate((x1,y1),-1)
        t2 = np.concatenate((x2,y2),-1)

        t = np.concatenate((t1,t2))
        np.random.shuffle(t)
        
        #save datapoints if save_as_try   
        if save_as_try is not None:
            try:
                stringdex = self.save_name.index('samp') + 4
                #stringdex_end = -1 #self.save_path.find()
            
                self.save_name = self.save_name[:stringdex] + f'{n}'
                
            except:
                self.save_name = self.save_name + f'_samp{n}'
                
            np.save(self.save_path + self.save_name + f'_try{save_as_try}.npy', t)
            
        return t
    
    def sample_4_gaussians(self, n, w = None, save_as_try=None):

        if w is None:
            w = [1,1,1,1]
        
        w = np.array(w)/np.sum(w)
        
        t = []
        for i in range(2):
            for j in range(2):
                t.append(np.random.multivariate_normal([-4 + 8 * i, -4 + 8 * j], [[1,0],[0,1]], int(n*w[2*i+j])))

        t = np.concatenate(t)
        np.random.shuffle(t)
        
        #save datapoints if save_as_try   
        if save_as_try is not None:
            try:
                stringdex = self.save_name.index('samp') + 4
                #stringdex_end = -1 #self.save_path.find()
            
                self.save_name = self.save_name[:stringdex] + f'{n}'
                
            except:
                self.save_name = self.save_name + f'_samp{n}'
                
            np.save(self.save_path + self.save_name + f'_try{save_as_try}.npy', t)
            
        return t
    
    def sample_3_rings(self, n, save_as_try=None):

        t = []

        n=n//3
        
        r1 = np.random.normal(4,1, [n,1])
        r2 = np.random.normal(4,1, [n,1])
        r3 = np.random.normal(4,1, [n,1])
        phi1 = np.random.uniform(0,2*np.pi, [n,1])
        phi2 = np.random.uniform(0,2*np.pi, [n,1])
        phi3 = np.random.uniform(0,2*np.pi, [n,1])

        x1 = r1 * np.cos(phi1) + 12
        x2 = r2 * np.cos(phi2) - 12
        x3 = r3 * np.cos(phi3) 
        y1 = r1 * np.sin(phi1)
        y2 = r2 * np.sin(phi2)
        y3 = r3 * np.sin(phi3)

        t1 = np.concatenate((x1/3,y1),-1)
        t2 = np.concatenate((x2/3,y2),-1)
        t3 = np.concatenate((x3/3,y3),-1)

        t = np.concatenate((t1,t2, t3))
        np.random.shuffle(t)
        
        #save datapoints if save_as_try   
        if save_as_try is not None:
            try:
                stringdex = self.save_name.index('samp') + 4
                #stringdex_end = -1 #self.save_path.find()
            
                self.save_name = self.save_name[:stringdex] + f'{n}'
                
            except:
                self.save_name = self.save_name + f'_samp{n}'
                
            np.save(self.save_path + self.save_name + f'_try{save_as_try}.npy', t)
            
        return t
    
'''
def drawSamples(n = 10000):
    scale = 1.
    mylambda = 0.01
    mu = 1.0
    sigma = 0.04
    nback = np.random.binomial(1,1-mylambda,n)
    back = np.random.exponential(scale,sum(nback))
    signal = np.random.normal(mu,sigma,n-sum(nback))
    x = np.concatenate([back,signal])
    x = np.expand_dims(x, axis=1)    
    return x
'''
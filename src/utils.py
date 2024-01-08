# define all the other stuff here

import numpy as np
import numpy.random as random

#########################################################################
### Quantileing methods giving n_dim arrays of ascending quants edges ### 
#########################################################################

def quantiles(base_data, n_quant, verbose = False, mode = 'density', rng = None):
    '''
    constructs quantiles from data of the real distribution
    '''
    
    assert mode in ['density', 'length']
        
    quants_list = []
    n_dim = base_data.shape[1]

    for d in range(n_dim):
        if mode == 'density':
            quants = []
            for i in range(1, int(n_quant)):
                quants.append(np.quantile(base_data[:,d], i/(n_quant)))    
        else:
            bounds = rng[d]  if rng is not None else (min(base_data[:,d]), max(base_data[:,d]))
            quants = np.linspace(*bounds, n_quant+1)

        quants_list.append(np.asarray(quants))

    if verbose:
        print(f'quant list: {quants_list}')
        
    return quants_list

def quantile_values(quantile_list, data):
    '''
    determines the relative number/density of points in each quantile in each dimension
    INPUT:
       quantile_list -- list: len(quantile_list) != n_dim, list elements are the qunatiles in that dimension
                data -- np.array: data.shape != (number of datapoints, n_dim)
    OUTPUT:
    np.array: (n_dim, len(quantile_list)+1) density of points in each quantile
    '''
    n_dim = len(quantile_list)
    quantvals_list = []
    
    for dim in range(n_dim):
        quantiles = quantile_list[dim]
        
        prev_val = 1
        quantvals = []
        
        n_total = data.shape[0]

        for cutoff in quantiles:
            val = ((data[:,dim] > cutoff).sum())/(n_total)
            quantval = prev_val - val
            prev_val = val
            quantvals.append(quantval)

        quantval = ((data[:,dim] > quantiles[-1]).sum())/(n_total)
        quantvals.append(quantval)
        quantvals_list.append(quantvals)

    return np.asarray(quantvals_list)

def quantile_values_via_hist(quantile_list, data, weights=None, embed=False):
    '''
    determines the relative number/density of points in each quantile with multidim quantiles
    INPUT:
       quantile_list -- list: len(quantile_list) != n_dim, list elements are the qunatiles in that dimension         
                data -- np.array: data.shape != (number of datapoints, n_dim)
    OUTPUT:
    np.array: n_dimensions of n_quantiles each
    '''
    
    n_dim = len(quantile_list)
    bins_data = [np.concatenate((np.finfo(np.float16).min, quantiles,  np.finfo(np.float16).max), 0) for qunatiles in quantile_list] if embed else quantile_list
                
    quant_hist, edges = np.histogramdd(sample = data, bins = bins_data, weights = weights, range = None)
    quant_hist = quant_hist/data.shape[0]
    return quant_hist

#######################
### Half quantiling ###
#######################

def genertateQunatileSplit(data, n_splits_total, data_borders = None, n_splits_current=0, **kwargs):
    '''
    Iterative method constructing equal weight regions by dividing into two in every dimension alternatingly
    INPUT:
    data: 2D array [n_points, n_dim]
          [[x_1, x_2, ... x_n],
            ...
           [x_1, x_2, ... x_n]]  
    n_splits_total: number of splits, will result in 2^n regions  
    data_borders: recursive flag  
    n_splits_current: recursive flag
    
    OUTPUT:
    np.array: with shape (number of splits i.e. 2^n, number of dimensions, 2 for upper and lower bound of box)
    '''
    
    if kwargs.get('verbose', False):
        print(f'{n_splits_current} splits, {len(data)} points')
    
    try:
        n_dim = data.shape[1]
    except:
        n_dim = 1
        data = data.reshape(-1,1)
        
    if n_splits_total == n_splits_current:
        return data_borders
        
    split_axis = n_splits_current%n_dim
    
    try: 
        if data_borders == None:
            data_borders = np.array([[[np.finfo(np.float16).min, 
                                       np.finfo(np.float16).max]]*n_dim])  
            
    except:
        pass
    
    med = np.median(data[:, split_axis])
    
    split1 = np.array(data[(data[:, split_axis] >  med)])
    split2 = np.array(data[(data[:, split_axis] <= med)])
        
    split1_borders = np.copy(data_borders)
    split1_borders[0,split_axis,0] = med
        
    split2_borders = np.copy(data_borders)
    split2_borders[0,split_axis,1] = med
    
    ret1 = genertateQunatileSplit(split1, n_splits_total, split1_borders, n_splits_current+1)
    ret2 = genertateQunatileSplit(split2, n_splits_total, split2_borders, n_splits_current+1)
    
    ret = np.concatenate((ret1, ret2), axis=0)
    
    return ret


def genertateQunatileSplit_sortonce(data, n_splits_total, sort_args = None, data_borders = None, n_splits_current=0):
    '''
    Iterative method constructing equal weight regions by dividing the argsort-arrays into two in every 
    dimension alternatingly. This why the data only needs to be sorted once!
    INPUT:
    data: 2D array [n_points, n_dim]
          [[x_1, x_2, ... x_n],
            ...
           [x_1, x_2, ... x_n]]  
    n_splits_total: number of splits, will result in 2^n regions  
    data_borders: recursive flag  
    n_splits_current: recursive flag
    
    OUTPUT:
    np.array: with shape (number of splits i.e. 2^n, number of dimensions, 2 for upper and lower bound of box)
    '''
    try:
        n_dim = data.shape[1]
    except:
        n_dim = 1
        data = data.reshape(-1,1)
        
    if n_splits_total == n_splits_current:
        return data_borders
    
    split_axis = n_splits_current%n_dim
    
    try: 
        if data_borders == None:
            data_borders = np.array([[[np.finfo(np.float16).min, 
                                       np.finfo(np.float16).max]]*n_dim])  
        if sort_args == None:     #do the first sorting --> only use the argsort-arguments afterwards
            sort_args = [np.argsort(data[:, i]) for i in range(n_dim)]
            print('genertateQunatileSplit_sortonce: Finished the sorting --> this should be fast from here')
    except:
        pass
    
    #determine the median on the split_axis
    sort_arg = sort_args[split_axis]
    half_idcs1 = sort_arg[:len(sort_arg)//2]
    half_idcs2 = sort_arg[len(sort_arg)//2:]
    med = data[half_idcs1[-1], split_axis]
    
    #build a mask to determine which values are <med in the dataset (and thus in the sorting arrays of the other dimensions)
    maskarr = np.zeros(len(data))
    maskarr[half_idcs1] = 1
    
    #determine the arguments that have  a 1 or a 0 and put them into new sorting arrays
    sort_args1 = [arg[maskarr[arg]==1] for arg in sort_args]
    #print(len(sort_args1), sort_args1[0].shape, maskarr.shape, maskarr.sum(), half_idcs1)
    sort_args2 = [arg[maskarr[arg]==0] for arg in sort_args]
    
    split1_borders = np.copy(data_borders)
    split1_borders[0,split_axis,1] = med
        
    split2_borders = np.copy(data_borders)
    split2_borders[0,split_axis,0] = med
    
    ret1 = genertateQunatileSplit_sortonce(data, n_splits_total, sort_args1, split1_borders, n_splits_current+1)
    ret2 = genertateQunatileSplit_sortonce(data, n_splits_total, sort_args2, split2_borders, n_splits_current+1)
    
    ret = np.concatenate((ret1, ret2), axis=0)
    
    return ret


def evaluateQunatileSplit(split_edges, data):
    '''
    
    INPUT:
    data: 2D array [n_points, n_dim]
          [[x_1, x_2, ... x_n],
            ...
           [x_1, x_2, ... x_n]] 
    split_edges: output of the genertateQunatileSplit function
    
    OUTPUT:
    np.array: n_dimensions of n_quantiles each
    '''
    try:
        n_dim = data.shape[1]
    except:
        n_dim = 1
        data = data.reshape(-1,1)
        
    outlist = np.ones(split_edges.shape[0])
    
    for i in range(0, split_edges.shape[0]):
        data_in_split = np.ones(data.shape[0])
        for dim in range(n_dim):
            data_in_split[(data[:, dim] <=  split_edges[i,dim,0])] = 0
            data_in_split[(data[:, dim] > split_edges[i,dim,1])] = 0
        
        outlist[i] = np.sum(data_in_split)
        
    #print(np.unique(outlist), np.sum(outlist))
    #u1, c1 = np.unique(data[:,0], return_counts = True)
    #u2, c2 = np.unique(data[:,1], return_counts = True)
    #print(u1[c1>2], c1[c1>2])#, u2[c2>2], c2[c2>2])
    #print(f'normalizing through division by {data.shape[0]}')
    
    return np.array(outlist)/data.shape[0]

#########################
### quantile measures ###
#########################

def quantile_MSE_sqrt(data, n_quant = None, n_dim = None, eval_single_dims = True, data_true = None):
    
    if n_quant is None:
        n_quant = data.shape[0]
    if n_dim is None:
        try:
            n_dim = data.shape[1]
        except:
            n_dim = 1
        
    if not eval_single_dims:
        n_quant = n_quant**n_dim
        
    p = 1/n_quant if data_true is None else data_true
    q = data
    
    tmp = p*(q - p)**2
    
    if eval_single_dims:
        MSE_sqrt = np.sqrt(np.sum(np.nan_to_num(tmp, nan = 0, posinf = 0, neginf = 0), 3))
    else:
        MSE_sqrt = np.sqrt(np.sum(np.nan_to_num(tmp, nan = 0, posinf = 0, neginf = 0), 
                                  tuple(range(-n_dim,0))))
    return MSE_sqrt

def quantile_PCT(data, n_quant = None, n_dim = None, eval_single_dims = True, data_true = None):
    
    if n_quant is None:
        n_quant = data.shape[0]
    if n_dim is None:
        try:
            n_dim = data.shape[1]
        except:
            n_dim = 1
        
    if not eval_single_dims:
        n_quant = n_quant**n_dim
        
    p = 1/n_quant if data_true is None else data_true
    q = data
    
    tmp = np.abs(q - p)
    
    if eval_single_dims:
        PCT = np.sum(np.nan_to_num(tmp, nan = 0, posinf = 0, neginf = 0), 3)
    else:
        PCT = np.sum(np.nan_to_num(tmp, nan = 0, posinf = 0, neginf = 0), 
                                  tuple(range(-n_dim,0)))
    return PCT

def quantile_chi2(data, n_quant = None, n_dim = None, eval_single_dims = True, data_true = None):
    
    if n_quant is None:
        n_quant = data.shape[0]
    if n_dim is None:
        try:
            n_dim = data.shape[1]
        except:
            n_dim = 1
        
    if not eval_single_dims:
        n_quant = n_quant**n_dim
    
    p = 1/n_quant if data_true is None else data_true
    q = data
    
    tmp = (p - q)**2/p
    
    if eval_single_dims:
        chi2 = np.sum(np.nan_to_num(tmp, nan = 0, posinf = 0, neginf = 0), 3)
    else:
        chi2 = np.sum(np.nan_to_num(tmp, nan = 0, posinf = 0, neginf = 0), tuple(range(-n_dim,0)))  
    return chi2

def quantile_JSD(data, n_quant = None, n_dim = None, eval_single_dims = True, data_true = None):
    #test absolute continuity
    if n_quant is None:
        n_quant = data.shape[0]
    if n_dim is None:
        try:
            n_dim = data.shape[1]
        except:
            n_dim = 1
    
    if not eval_single_dims:
        n_quant = n_quant**n_dim
    
    p = 1/n_quant if data_true is None else data_true
    q = data
    
    tmp1 = p*np.log(2*p/(p+q)) 
    tmp2 = q*np.log(2*q/(p+q))
    tmp = np.nan_to_num(tmp1, nan = 0, posinf = 0, neginf = 0) + np.nan_to_num(tmp2, nan = 0, posinf = 0, neginf = 0)
    
    if eval_single_dims:
        chi2 = 0.5*np.sum(tmp, 3)
    else:
        chi2 = 0.5*np.sum(tmp, tuple(range(-n_dim,0)))  
    return chi2
    
'''
from ot import dist, emd2
from scipy.stats import wasserstein_distance, entropy
from scipy.spatial.distance import jensenshannon
def divergence_measure(xs, xt, minbin=-8, maxbim=8, nbin=32, dist = 'kld'):
    if dist == 'emd_2d':
        xs = xs[0:1000]
        xt = xt[0:1000]
        n = xs.shape[0]
        a, b = np.ones((n,)) / n, np.ones((n,)) / n
        M = dist(xs, xt)
        M /= M.max()
        return emd2(a,b,M)
    
    
    if dist == 'emd_proj':
        out = 0
        for i in range(0, xs.shape[1]):
            wasserstein_distance(xs[:,i], xt[:,i])
        return out
    
    if dist == 'rand':
        return random.random()
    
    
    if dist == 'kld_proj':
        out = 0
        for i in range(0, xs.shape[1]):
            hist_a = np.histogram(xs[:,i], density=True, bins=nbin, range=(minbin,maxbim))[0]
            hist_b = np.histogram(xt[:,i], density=True, bins=nbin, range=(minbin,maxbim))[0]
            hist_b = np.where(hist_b == 0.0, 1e-12, hist_b)
            hist_a = np.where(hist_a == 0.0, 1e-12, hist_a)
            out += entropy(hist_a, hist_b)
        return out
    
    
    if dist == 'jsd_proj':
        out = 0
        for i in range(0, xs.shape[1]):
            hist_a = np.histogram(xs[:,i], density=True, bins=nbin, range=(minbin,maxbim))[0]
            hist_b = np.histogram(xt[:,i], density=True, bins=nbin, range=(minbin,maxbim))[0]
            hist_b = np.where(hist_b == 0.0, 1e-12, hist_b)
            hist_a = np.where(hist_a == 0.0, 1e-12, hist_a)
            out += jensenshannon(hist_a, hist_b)
        return out
   
    if dist == 'kld_2d':
        print('Rethink this: does this make sense?')
        #edge_data = [] for i in range(0, xs.shape[1]):
        #    edge_data.append((minbin,maxbim)) hist_a = np.histogramdd(xs, density=True, bins=nbin, 
        #range=edge_data)[0] hist_b = np.histogramdd(xt, density=True, bins=nbin, range=edge_data)[0] hist_b = 
        #np.where(hist_b == 0.0, 1e-9, hist_b) hist_a = np.where(hist_a == 0.0, 1e-9, hist_a) print(hist_a) 
        #print(hist_b)
        return KLdivergence(xs, xt)
   
    if dist == 'jsd_2d':
        print('Rethink this: does this make sense?')
        #edge_data = [] for i in range(0, xs.shape[1]):
        #    edge_data.append((minbin,maxbim)) hist_a = np.histogramdd(xs, density=True, bins=nbin, 
        #range=edge_data)[0] hist_b = np.histogramdd(xt, density=True, bins=nbin, range=edge_data)[0] hist_b = 
        #np.where(hist_b == 0.0, 1e-9, hist_b) hist_a = np.where(hist_a == 0.0, 1e-9, hist_a)
        return 0.5*(KLdivergence(xs, xt) + KLdivergence(xt, xs))
'''
    
'''def twoGaussianUnb(x, mean1, sigma1, mean2, sigma2, A1):
    badvalue = 1e-300
    smallestdiv = 1e-10
    ret1 = 0
    ret2 = 0
    if sigma1 < smallestdiv or sigma2 < smallestdiv:
        ret1 = badvalue
        ret2 = badvalue
    else:
        d1 = (x-mean1)/sigma1
        d12 = d1*d1
        ret1 = A1/(np.sqrt(2*np.pi)*sigma1)*np.exp(-0.5*d12)
        d2 = (x-mean2)/sigma2
        d22 = d2*d2
        ret2 = (1-A1)/(np.sqrt(2*np.pi)*sigma2)*np.exp(-0.5*d22)
    return (ret1 + ret2) 
'''
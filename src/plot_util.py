#from iminuit import Minuit
#from probfit import UnbinnedLH, gaussian

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from utils import *

def ganplify_plot(n_GANed, metric_quantvals_fake, metric_quantvals_train, n_sampled, metric_quantvals_true, dim, 
                  n_training_samples, n_quant_str, y_label_str, save_str):
    '''
    plotting routine for ganplify plot
    INPUT:
    dim - single dimension if data is evaluated for single dimensions, else slice(None)
    '''
    color_list1 = ['turquoise', 'dodgerblue', 'blue', 'midnightblue', 'indigo', 'darkmagenta', 'magenta']
    color_list2 = ['orange', 'orangered', 'tomato', 'red', 'maroon', 'magenta', 'darkmagenta', 'indigo', 'midnightblue']
    
    color_list1 = color_list1[1:]
    color_list2 = [color_list2[i] for i in [1,4,6]]
    
    linestyle_list = 5*['-', '--']
    linewidth_list = 10*[1.5]
    
    x_shift2 = 1.0
    font_s = 25
    
    label_pos_gan = 0
    label_pos_train_sample = 0
    label_pos_true_samples = 0
    
    figSide = plt.figure(figsize=(8,8))
    axS1 = figSide.add_subplot(1,1,1)
    axS1.set_xscale('log')
    axS1.set_yscale('log')
    #axS1.set_ylim([2e-3, 3e-1])
    axS1.set_xlim([np.min(n_GANed)*0.8, np.max(n_GANed)*1.5])

    
    #plot the GANed results
    if dim == 'all':
        mean_metric_quantvals_fake = np.mean(metric_quantvals_fake[:,:], 0)
        std_metric_quantvals_fake = np.std(metric_quantvals_fake[:,:], 0)
    else:
        mean_metric_quantvals_fake = np.mean(metric_quantvals_fake[:,:,dim], 0)
        std_metric_quantvals_fake = np.std(metric_quantvals_fake[:,:,dim], 0)
    
    pStra = axS1.errorbar(x=n_GANed, 
                          y=mean_metric_quantvals_fake, 
                          xerr=None, 
                          yerr=std_metric_quantvals_fake, 
                          marker='.', 
                          linestyle=linestyle_list[0], 
                          linewidth=linewidth_list[0],  
                          label='GAN',
                          color=color_list2[0])    

    axS1.text(n_GANed[label_pos_gan],
              mean_metric_quantvals_fake[label_pos_gan],
              ' GAN', 
              horizontalalignment='left',
              verticalalignment='bottom', 
              color=color_list2[0], fontsize=font_s)   
    
    #plot training sample results
    if dim == 'all':
        mean_metric_quantvals_train = np.mean(metric_quantvals_train[:,0], 0)
        std_metric_quantvals_train = np.std(metric_quantvals_train[:,0], 0)
    else:
        mean_metric_quantvals_train = np.mean(metric_quantvals_train[:,0,dim], 0)
        std_metric_quantvals_train = np.std(metric_quantvals_train[:,0,dim], 0)
    
    pStra = axS1.errorbar(x=(n_GANed[0]*0.1, n_GANed[-1]*10),
                          y=(mean_metric_quantvals_train, mean_metric_quantvals_train),
                          linestyle=linestyle_list[0], 
                          linewidth=linewidth_list[0], 
                          label='sample',
                          color=color_list1[0])
    
    pStra = axS1.fill_between((n_GANed[0]*0.1, n_GANed[-1]*10),
                              mean_metric_quantvals_train + std_metric_quantvals_train,
                              mean_metric_quantvals_train - std_metric_quantvals_train,
                              label='fit',
                              color=color_list1[0],
                              alpha=0.2,
                              hatch='/')

    axS1.text(n_GANed[label_pos_train_sample],
              mean_metric_quantvals_train,
              ' sample', 
              horizontalalignment='left',
              verticalalignment='bottom', 
              color=color_list1[0], 
              fontsize=font_s)       

    #lastly plot the results for different amounts of sampled data
    if dim == 'all':
        mean_metric_quantvals_true = np.mean(metric_quantvals_true[:,:], 0)
        std_metric_quantvals_true = np.std(metric_quantvals_true[:,:], 0)
    else:
        mean_metric_quantvals_true = np.mean(metric_quantvals_true[:,:,dim], 0)
        std_metric_quantvals_true = np.std(metric_quantvals_true[:,:,dim], 0)
        
    for k, n in enumerate(n_sampled):
        pStra = axS1.errorbar(x=(n_GANed[0]*0.1, n_GANed[-1]*10),
                              y=(mean_metric_quantvals_true[k], mean_metric_quantvals_true[k]),
                              linestyle=linestyle_list[1], 
                              linewidth=1, 
                              label='{:d} samples'.format(n),
                              color=color_list1[0])
        label_pos = 0
        axS1.text(n_GANed[label_pos_true_samples],
                  mean_metric_quantvals_true[k],
                  ' {:d}'.format(n), 
                  horizontalalignment='left',
                  verticalalignment='bottom', 
                  color=color_list1[0], 
                  fontsize=font_s)  

    axS1.text(0.5, 0.91,
              n_quant_str + ' quantiles', horizontalalignment='left',verticalalignment='bottom', 
              color='k', fontsize=font_s, transform=axS1.transAxes)
    axS1.text(0.5, 0.82, 
              f'{n_training_samples} data points', horizontalalignment='left',verticalalignment='bottom', 
              color='k', fontsize=font_s, transform=axS1.transAxes)
    
    axS1.yaxis.set_major_locator(mtick.LogLocator(base=10, subs=range(0,11,2)))
    axS1.yaxis.set_minor_locator(mtick.LogLocator(base=10, subs=range(100)))
    plt.setp(axS1.get_yminorticklabels(), visible=False);

    for tick in axS1.xaxis.get_major_ticks():
        tick.label.set_fontsize(font_s) 
    for tick in axS1.yaxis.get_major_ticks():
        tick.label.set_fontsize(font_s) 

    #axS1.legend()
    #plt.subplots_adjust(left=0.18, right=0.95, top=0.95, bottom=0.18)
    
    formatter = mtick.FormatStrFormatter('%.3f')
    #formatter = mtick.StrMethodFormatter('{x:,.2f}')   
    
    #axS1.legend()
    plt.subplots_adjust(left=0.22, right=0.95, top=0.95, bottom=0.18)
    axS1.get_yaxis().set_major_formatter(formatter)

    axS1.set_xlabel('number GANed', fontsize=font_s) 
    
    if dim == 0:
        axS1.set_ylabel(y_label_str, fontsize=font_s, labelpad=-5) 


   # figSide.suptitle(name , fontsize=font_s)
    plt.savefig(save_str)        


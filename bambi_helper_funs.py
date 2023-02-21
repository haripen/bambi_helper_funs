#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#Created on Sun Mar 24 12:19:33 2019
#
#@author: haraldpenasso

#%% FUNCTIONS TO LOAD
#%%------------------------------------------------------------------------
# Test Function
def hello():
    print('!!! Hello World !!!')
#--------------------------------------------------------------------------
#%% ---------------------------------------------------------------------------
# f_test
# -----------------------------------------------------------------------------
# DESCRIPTION
def f_test(a,b):
    # idata_group is either the prior or posterior group
    import scipy.stats
    import numpy as np
    f = np.var(a,ddof=1)/np.var(b,ddof=1) # consider returning f too
    nun = a.size-1
    dun = b.size-1
    p = 1 - scipy.stats.f.cdf(f,nun, dun)
    return p
# -----------------------------------------------------------------------------
#%% ---------------------------------------------------------------------------
# summary_stats
# -----------------------------------------------------------------------------
# DESCRIPTION
def summary_stats(bf_df,effs,bf_pri_prob,label,rope = [-0.1,0.1],esml = [0.2,0.5,0.8],hdi_prob=0.95):
    import pandas as pd
    import arviz as az
    hdi  = az.hdi(effs, hdi_prob=hdi_prob)
    return pd.DataFrame(
    {
    'bf01'                : bf_df.loc['bf01',:].to_numpy(),
    'pc_EQ'               : 100*bf_df.loc['pH0gD',:].to_numpy(),
    'neg_pc_large'        : 100*(effs <= -esml[2]).mean().to_numpy(),
    'neg_pc_medium'       : 100*(((effs  > -esml[2]) * (effs <= -esml[1])).mean()).to_numpy(),
    'neg_pc_small'        : 100*(((effs  > -esml[1]) * (effs <= -esml[0])).mean()).to_numpy(),
    'neg_pc_below_small'  : 100*(effs < -esml[0]).mean().to_numpy(),
    'pc_in_rope'          : 100*(((effs<rope[1])*(effs>=rope[0])).mean()).to_numpy(),
    'hdi95_lower'         : hdi.sel({'hdi':'lower'})['scaled_value_sigma'].to_numpy()*1,
    'mean'                : effs.mean(),
    'hdi95_higher'        : hdi.sel({'hdi':'higher'})['scaled_value_sigma'].to_numpy()*1,
    'pos_pc_above_small'  : 100*(effs > esml[0]).mean().to_numpy(),
    'pos_pc_small'        : 100*(((effs  < esml[1]) * (effs >= esml[0])).mean()).to_numpy(),
    'pos_pc_medium'       : 100*(((effs  < esml[2]) * (effs >= esml[1])).mean()).to_numpy(),
    'pos_pc_large'        : 100*(effs >= esml[2]).mean().to_numpy(),
    'pc_DIFF'             : 100*bf_df.loc['pH1gD',:].to_numpy(),
    'bf10'                : bf_df.loc['bf10',:].to_numpy(),
    'p_pri_reject_diff'   : 100*bf_pri_prob[0],
    'p_pri_accept_diff'   : 100*bf_pri_prob[1]
    },index=[label]).astype(float).transpose().round(2)
# -----------------------------------------------------------------------------
#%% ---------------------------------------------------------------------------
# sort_dict_by_mean
# -----------------------------------------------------------------------------
# DESCRIPTION
def sort_dict_by_mean(di):
    import numpy as np
    return {k: v for k, v in sorted(di.items(), key=lambda item: np.mean(item[1]))}
# -----------------------------------------------------------------------------
#%% ---------------------------------------------------------------------------
# sns_to_hex
# -----------------------------------------------------------------------------
# DESCRIPTION
def sns_to_hex(sea_col):
    import numpy as np
    return '#%02x%02x%02x' % tuple([int(i) for i in (np.array(sea_col)*255).round()])
# -----------------------------------------------------------------------------
#%% ---------------------------------------------------------------------------
# sel_data_from_az
# -----------------------------------------------------------------------------
# DESCRIPTION
def sel_data_from_az(idata,df,h0_dict,h1_dict=None, kind = None, mode = None, kde_bandwidth = 0.001, random_seed=1234):
    from scipy.stats import gaussian_kde
    import numpy as np
    if kind == 'null' and h1_dict==None:
        # Loop through key-item pairs and gett boolian array for selection
        idx_bool_da0 =  idx_from_dict(h0_dict,df)
        h1_dat = az_isel_scaled_value(idata,idx_bool_da0)['posterior']
        h0_dat = az_isel_scaled_value(idata,idx_bool_da0)['posterior']

        # center h0_dat _mean to 0
        h0_dat['scaled_value_mean'] = h0_dat.scaled_value_mean-h0_dat.scaled_value_mean.mean()
        # resample h0_dat distribiutions randomly
        obs = list() 
        for c in h0_dat.chain:
            mn = list()
            for i,o in enumerate(h0_dat.scaled_value_obs):
                vals = h0_dat['scaled_value_mean'].sel(scaled_value_obs=int(o),chain=int(c)).values
                mn.append(gaussian_kde(vals,kde_bandwidth).resample(len(vals),random_seed+i)[0])
            obs.append(mn)
        h0_dat['scaled_value_mean'].values = np.swapaxes(obs,1,2) 
        sig = list()
        for i,c in enumerate(h0_dat.chain):
            vals = h0_dat['scaled_value_sigma'].sel(chain=int(c)).values
            sig.append(gaussian_kde(vals,kde_bw).resample(len(vals),random_seed+i)[0])
        h0_dat['scaled_value_sigma'].values=sig
    if kind == 'effect_size' and h1_dict != None:
        idx_bool_da0 =  idx_from_dict(h0_dict,df)
        idx_bool_da1 =  idx_from_dict(h1_dict,df)
        h0_dat = az_isel_scaled_value(idata,idx_bool_da0)['posterior']
        h1_dat = az_isel_scaled_value(idata,idx_bool_da1)['posterior']
        
    if kind == "bayes_factor":
        if mode == "global":
            col_H = list(h0_dict.keys())[0]
            col_H_lvl_h0 = h0_dict[col_H]
            col_H_lvl_h1 = h1_dict[col_H]
            h0_dat = (idata.posterior.stack(draws=("chain", "draw"))[col_H]).sel({col_H+'_dim':col_H_lvl_h0})
            h1_dat = (idata.posterior.stack(draws=("chain", "draw"))[col_H]).sel({col_H+'_dim':col_H_lvl_h1})        
        elif mode == "ids" or mode == "tests":
            col_H = list(h0_dict.keys())[0]
            col_A = list(h0_dict.keys())[1]
            col_A_lvl =    h0_dict[col_A]
            col_H_lvl_h0 = h0_dict[col_H]
            col_H_lvl_h1 = h1_dict[col_H]
            h0_dat = (idata.posterior.stack(draws=("chain", "draw"))[col_A+':'+col_H]).sel({col_A+':'+col_H+'_dim':col_A_lvl+', '+col_H_lvl_h0})
            h1_dat = (idata.posterior.stack(draws=("chain", "draw"))[col_A+':'+col_H]).sel({col_A+':'+col_H+'_dim':col_A_lvl+', '+col_H_lvl_h1})
            
    return h0_dat, h1_dat
# -----------------------------------------------------------------------------
#%% ---------------------------------------------------------------------------
# az_effect_size
# -----------------------------------------------------------------------------
# DESCRIPTION
def effect_size(h0_dat,h1_dat,sig_diff=0.05):
    n_h0 = h0_dat['scaled_value_sigma'].size
    n_h1 = h1_dat['scaled_value_sigma'].size
    n = n_h1 + n_h0
    if f_test(h0_dat['scaled_value_sigma'],h1_dat['scaled_value_sigma'])<sig_diff:
        print("Computing corrected Glass's delta")
    else:
        print("Computing corrected Hedge's g")
        sd_pooled = ( ( (n_h0-1)*h0_dat['scaled_value_sigma']**2 + (n_h1-1)*h1_dat['scaled_value_sigma']**2) / (n_h0+n_h1-2) )**(1/2)
    diff_of_means = h1_dat['scaled_value_mean'].mean(axis=2).values-h0_dat['scaled_value_mean'].mean(axis=2).values
    return (diff_of_means/sd_pooled) * ((n-3)/(n-2.25))*((n-2)/n)**(1/2)
# -----------------------------------------------------------------------------

#%% ---------------------------------------------------------------------------
# bambi2idata4ppc
# -----------------------------------------------------------------------------
# DESCRIPTION
def bambi2idata4ppc(model,result,draws=6000,random_seed = 1234):
    model.predict(result, inplace=True, kind="mean")
    idata = model.predict(result, inplace=False, kind="pps")
    idata.extend( model.prior_predictive(draws=draws,random_seed=random_seed) )
    return idata
# -----------------------------------------------------------------------------
#%% ---------------------------------------------------------------------------
# az_isel_scaled_value
# -----------------------------------------------------------------------------
# DESCRIPTION
def az_isel_scaled_value(idata,sel):
    idata = idata.isel(scaled_value_obs=sel).isel(scaled_value_dim_0=sel)
    return idata
# -----------------------------------------------------------------------------
#%% ---------------------------------------------------------------------------
# ppc_bambi
# -----------------------------------------------------------------------------
# DESCRIPTION
# could remove subset option using new intersecti fun 
def ppc_bambi(idata,df,var_names=[],subset=[],check_names = ['prior','posterior'],n_pp_samp=[],colors=['#5AA36A','k','orange'], kind='cumulative',random_seed = 1234):
    import matplotlib.pyplot as plt
    import arviz as az
    if len(var_names)>0:
        d = dict()
        n_lvls = list()
        for var in var_names:
            d[var] = list(df[var].unique())
            n_lvls.append(len(d[var]))
        n_lvls = sum(n_lvls)
        if subset == []:
            n_subs = 1
        else:
            subset_levels = df[subset].unique().to_list()
            n_subs = len(subset_levels)
        n_checks = len(check_names)

        fig,axs = plt.subplots(n_lvls*n_subs,n_checks,figsize=(6*n_checks,3*n_lvls*n_subs),sharex='col',sharey='row')
        r = 0
        show_leg = True
        for key in d:
            for item in d[key]:
                if subset == []:
                    sel = idx_from_dict({key:item},df)
                    for c, check in enumerate(check_names):
                        print('plotting ... ax '+str(r)+','+str(c)+' - '+key+' '+str(item)+' - '+check)
                        if n_pp_samp != []:
                            az.plot_ppc(az_isel_scaled_value(idata,sel), group=check,
                                        random_seed=random_seed, legend=show_leg, ax=axs[r,c],
                                        colors=colors, kind=kind, num_pp_samples=n_pp_samp, show= False)
                        else:
                            az.plot_ppc(az_isel_scaled_value(idata,sel), group=check,
                                        random_seed=random_seed, legend=show_leg, ax=axs[r,c],
                                        colors=colors, kind=kind, show= False)
                        if r < n_lvls-1:
                            axs[r,c].set_xlabel(None)
                        if r == 0:
                            axs[r,c].legend(loc='upper left')
                        axs[r,c].grid()
                        axs[r,c].set_title(key+' - '+str(item),fontsize=14)
                    r = r + 1
                    if r > 0:
                        show_leg = False
                else:
                    for sub in subset_levels:
                        sel = idx_from_dict({key:item,subset:sub},df)
                        for c, check in enumerate(check_names):
                            print('plotting ... ax '+str(r)+','+str(c)+' - '+key+' '+str(item)+' : '+subset+' '+str(sub)+' - '+check)
                            if n_pp_samp != []:
                                az.plot_ppc(az_isel_scaled_value(idata,sel), group=check,
                                            random_seed=random_seed, legend=show_leg, ax=axs[r,c],
                                            colors=colors, kind=kind, num_pp_samples=n_pp_samp, show= False)
                            else:
                                az.plot_ppc(az_isel_scaled_value(idata,sel), group=check,
                                            random_seed=random_seed, legend=show_leg, ax=axs[r,c],
                                            colors=colors, kind=kind, show= False)
                            if r < n_lvls*n_subs-1:
                                axs[r,c].set_xlabel(None)
                            if r == 0:
                                axs[r,c].legend(loc='upper left')
                            else:
                                show_leg = False
                            axs[r,c].grid()
                            axs[r,c].set_title(key+' '+str(item)+' : '+subset+' '+str(sub),fontsize=14)
                        r = r + 1
                        if r > 0:
                            show_leg = False
        fig.tight_layout()
    else:
        fig,axs = plt.subplots(1,2,figsize=(12,3),sharey='row')
        for c, check in enumerate(check_names):
            print('plotting ... ax '+str(0)+','+str(c)+' - '+'global'+' '+'global'+' - '+check)
            if n_pp_samp != []:
                az.plot_ppc(idata, group=check,random_seed=random_seed, ax=axs[c],
                            colors=colors, kind=kind, num_pp_samples=n_pp_samp, show= False)
            else:
                az.plot_ppc(idata, group=check,random_seed=random_seed, ax=axs[c],
                            colors=colors, kind=kind, show= False)
            axs[c].legend(loc='upper left')
            axs[c].grid()
            axs[c].set_title('global',fontsize=14)
    return fig,axs
# -----------------------------------------------------------------------------
#%% ---------------------------------------------------------------------------
# az_plot_contrast
# -----------------------------------------------------------------------------
# DESCRIPTION
def az_plot_contrast(idata,df,h0_dict,h1_dict,t2,mode=None,ref_val=0, hdi = [-0.1,0.1],sav_plot_q=True):    
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    import arviz as az
    """
    """
    bf_d0, bf_d1 = sel_data_from_az(idata,df,h0_dict,h1_dict,kind="bayes_factor",mode=mode)
    es_d0, es_d1 = sel_data_from_az(idata,df,h0_dict,h1_dict,kind="effect_size")
    bf_df = bayes_factor(bf_d0,bf_d1)
    effs  = effect_size(es_d0, es_d1)

    # Plot
    col_H = list(h0_dict.keys())[0]
    col_H_lvl_h0 = h0_dict[col_H]
    col_H_lvl_h1 = h1_dict[col_H]
    fig,axs = plt.subplots(1,3,figsize=(12,3))
    sns.kdeplot({col_H_lvl_h1+'$_\Delta$':bf_d1,col_H_lvl_h0+'$_\Delta$':bf_d0}, fill=True, ax=axs[0])
    sns.despine()
    axs[0].set_title(t2+' BF$_{\mathrm{1,0}}$ = '+str(np.round(bf_df.loc['bf10',:][0],2))+', '+
                        'BF$_{\mathrm{0,1}}$ = '+str(np.round(bf_df.loc['bf01',:][0],2)),size=14)
    axs[0].set_xlabel(r'declined < 0 < improved'
                      '\n'
                      "scaled value",size=14)
    axs[0].set_ylabel('density',size=14)
    
    az.plot_posterior(effs, ref_val=ref_val, rope=(hdi[0],hdi[1]),hdi_prob=0.95, ax=axs[1]);
    axs[1].set_title(t2+' '+col_H_lvl_h1+'$_\Delta$ - '+col_H_lvl_h0+'$_\Delta$ contrast',size=14)
    axs[1].set_xlabel(r'declined < 0 < improved'
                      '\n'
                      "effect size",size=14)
    
    _,p05,p95,ax_ppbf = prior_prob_BF(bf_d0,bf_d1,fig=fig,ax=axs[2])

    plt.gcf().tight_layout()
    if sav_plot_q:
        flatten = lambda l: [item for sublist in l for item in sublist]
        fn = "_".join([ *set( flatten([[k,str(h0_dict[k])] for k in h0_dict.keys()])+flatten([[k,str(h1_dict[k])] for k in h1_dict.keys()]) ) ])
        plt.savefig(fn+'.pdf')
    plt.show()
    return bf_df, effs, (p05,p95)
# -----------------------------------------------------------------------------
#%% ---------------------------------------------------------------------------
# bayes_factor
# -----------------------------------------------------------------------------
def bayes_factor(da0, da1, prior_sim=0.5, prior_diff=0.5, kde_bandwidth = 0.001):
    import numpy as np
    from scipy.stats import gaussian_kde #norm
    import pandas as pd
    """
    """
    
    # Compute the mean and standard deviation of the two input DataArrays
    mean0 = da0.mean().values
    #std1 = da0.std().values
    mean1 = da1.mean().values
    #std1 = da1.std().values
    
    # Compute the likelihood for the similarity hypothesis
    # sd and mean method
    #odds_sim = (norm(mean1, std1).pdf(mean2) * norm(mean2, std2).pdf(mean1))
    # KDE methods
    # sklearn may provides more kernels
        #from sklearn.neighbors import KernelDensity as kde
        #np.exp(kde(kernel='epanechnikov',bandwidth=0.005).fit([[i] for i in da1.values]).score_samples([[mean2]]))
    odds_sim = gaussian_kde(da0,kde_bandwidth).pdf(mean1) * gaussian_kde(da1,kde_bandwidth).pdf(mean0)
    
    # Compute the likelihood for the difference hypothesis
    #odds_diff = (norm(mean1, std1).pdf(mean1) * norm(mean2, std2).pdf(mean2))
    odds_diff = gaussian_kde(da0,kde_bandwidth).pdf(mean0) * gaussian_kde(da1,kde_bandwidth).pdf(mean1)
    
    # Compute the probability for similarity
    p_sim = (odds_sim * prior_sim) / (odds_sim * prior_sim + odds_diff * prior_diff)
    
    # Compute the probability for difference
    p_diff = (odds_diff * prior_diff) / (odds_sim * prior_sim + odds_diff * prior_diff)
    
    # Compute the Bayes factor for similarity
    BF_diff = p_diff/p_sim
    
    # Compute the Bayes factor for difference
    BF_sim  = 1/BF_diff
    
    d = {'pH1':prior_diff,
         'pH0':prior_sim,
         'pH1gD':p_diff,
         'pH0gD':p_sim,
         'posterior_odds': odds_diff,
         'prior_odds': odds_sim,
         'bf10': BF_diff,
         'bf01': BF_sim }
    
    return pd.DataFrame(d,index=['Test_H0_on_similarity']).astype(float).transpose()

# -----------------------------------------------------------------------------
#%% ---------------------------------------------------------------------------
# myROC
# -----------------------------------------------------------------------------
def myROC(nums,cats,pos_cat=True,neg_cat=False,steps=1000):
    """
    Receiver operating characteristic from false-positive and true positive rates
    #vectorized
    
    INPUT
    ------
    nums    ... numpy array of numbers, dim: 1. The data
    cats    ... numpy array of boolean, dim: 1. The reference categories: 
                - True:  Responder (or as defined in pos_cat)
                - False: Non-Responder (or as defined in neg_cat)
                
    OPTIOAL
    ------
    pos_cat ... single str or boolian defining responder
    neg_cat ... single str or boolian defining non-responder
    steps   ... int. Number+2 of equally spaced thresholds for calculating tpr and fpr
    
    OUTPUT
    ------
    tpr     ... numpy array of floats, sensitivity or true positive rates. Each value represents a different threshold.
    fpr     ... numpy array of floats, float, 1-specificity of false positive rates Each value represents a different threshold.
    auc     ... single float, maximum of area under/above the curve instead of deflecting the data
    th_max  ... single float, the data-threshold at the Youden index (or true skill statistic) maximum.
    J_max   ... single float, the Youden index maximum
    
    (c) Harald Penasso 01/2023
    ToDo: Input checks
    """
    import numpy as np
    step_size = (nums.max()-nums.min())/steps # threshold step size in data units
    P = cats==pos_cat # condition positive (P) the number of real positive cases in the data
    N = cats==neg_cat # condition negative (N) the number of real negative cases in the data
    ths = np.expand_dims(np.arange(nums.min(),nums.max()+2*step_size,step_size),axis=0) # create thresholds
    nums = np.expand_dims(nums,axis=1) # prepare input for vactorization
    # true positive (TP) A test result that correctly indicates the presence of a condition or characteristic / P
    tpr = (np.expand_dims(P,axis=1) * (ths > nums)).sum(axis=0) / P.sum() # sensitivity
    # true negative (TN) A test result that correctly indicates the absence of a condition or characteristic / N
    fpr = (1-(np.expand_dims(N,axis=1) * ~(ths > nums)).sum(axis=0) / N.sum()) # 1-specificity
    auc = np.trapz(fpr,tpr) # calculate the area under the curve
    if auc < 0.5:
        auc = 1-auc # deflect area
    J = np.abs(np.array(tpr)-np.array(fpr)) # Youden-index or true skill statistic
    return tpr, fpr, auc, ths[0,np.argmax(J)], J[np.argmax(J)]
# -----------------------------------------------------------------------------
#%% ---------------------------------------------------------------------------
# get_prior_probs
# -----------------------------------------------------------------------------
# DESCRIPTION
def get_prior_probs(vals,prior_probs):
    from scipy.interpolate import CubicSpline
    cs = CubicSpline(vals, prior_probs)
    return [cs(0.05),cs(0.95)]
# -----------------------------------------------------------------------------

#%% ---------------------------------------------------------------------------
# prior_prob_BF
# -----------------------------------------------------------------------------
# DESCRIPTION
def prior_prob_BF(da0, da1, fig, ax, stepsize = 0.001,figsize=(4.5,4.5)):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    p_pris = np.arange(10**(-16),1+stepsize,stepsize)
    vals = [0]
    for p_pri in p_pris:
        vals.append(bayes_factor(da0, da1, prior_sim=1-p_pri, prior_diff=p_pri).loc['pH1gD',:].to_list()[0])
    p_pris = np.insert(p_pris,0,0)
    bf = bayes_factor(da0, da1, prior_sim=0.5, prior_diff=0.5).loc['bf10',:].to_list()[0]
    p05,p95 = get_prior_probs(vals,p_pris)
    if (p95 > 1):
        p95 = 1.0
    elif (p95 < 0):
        p95 = 0.0
    if (p05 > 1):
        p05 = 1.0
    elif (p05 < 0):
        p05 = 0.0
    #fig,ax = plt.subplots(1,1,figsize=figsize)
    ax.plot(p_pris,vals,zorder=100,linewidth=3,alpha=0.8)
    ax.plot((p05,p95),(0.05,0.95),'X', markersize=10)
    ax.fill_between((0,1),(1,1),(0.95,0.95),color='k',alpha=0.2,linewidth=0,zorder=1)
    ax.fill_between((0,1),(0,0),(0.05,0.05),color='k',alpha=0.2,linewidth=0,zorder=1)
    if ~np.isnan(p95):
        ax.text(0.2,0.85,'Prior s.t. Post > 0.95 = '+str(np.round(p95,2)),zorder=1000)
    if ~np.isnan(p05):
        ax.text(p05+0.05,0.07,'Prior s.t. Post < 0.05 = '+str(np.round(p05,2)),zorder=1000)
    ax.set_xlabel('Prior prob. dists differ')
    ax.set_ylabel('Post. prob. dists differ')
    #ax.set_title(key+' BF ('+lvl_h0+'/'+lvl_h1+') = '+str(np.round(bf,2)))
    ax.grid()
    sns.despine(right=True, top=True)
    ax.set_xlim((0,1))
    ax.set_ylim((0,1))
    plt.tight_layout()
    #plt.savefig('prior_prob_BF_'+key+'.pdf')
    #plt.show()
    return bf,p05,p95,ax
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#%% ---------------------------------------------------------------------------
# scaleback
# -----------------------------------------------------------------------------
# DESCRIPTION
def scaleback(t,df):
    s = df[df['test']==t].scaler.mean()
    if t == 'HealthVAS':
        s = s/10
        unit = ' [cm]'
        t_new = 'health VAS'
    elif t == 'WT2min':
        s = s/100
        unit = r' [$\times10^2$ m]'
        t_new = 'walk 2 min'
    elif t == 'TUG':
        unit = ' [s]'
        t_new = 'TUG'
    elif t == 'WT10m':
        unit = ' [s]'
        t_new = 'walk 10 m'
    elif t == 'FSST':
        unit = ' [s]'
        t_new = t 
    elif t == 'Velocity':
        unit = r' [km/h]'
        t_new = 'gait speed'
        s=s
    elif t == 'StanceTimeAL':
        unit = r' [$\times10^{-2}$ s]'
        t_new = 'stance time AL'
        s=s*10
    elif t == 'StepLengthUL':
        unit = r' [$\times10^1$ cm]'
        t_new = 'step length UL'
        s=s/10
    return s,unit,t_new
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#%% ---------------------------------------------------------------------------
# idx_from_dict
# -----------------------------------------------------------------------------
# DESCRIPTION
def idx_from_dict(d,df):
    """
    d ... dict, each key corresponds to a df column and each value to a cell entry of that column
    returns combined matches as boolian array
    """
    for i,k in enumerate(d):
        if i == 0:
            idx_bool = df[k].astype('str')==str(d[k])
        else:
            idx_bool = idx_bool * (df[k].astype('str')==str(d[k]))
    return idx_bool
# -----------------------------------------------------------------------------
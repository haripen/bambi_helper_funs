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
# get_data_for_effect_size
# -----------------------------------------------------------------------------
# DESCRIPTION
def get_data_for_effect_size(idata,df,main_key,sub_key=None):
    
    if sub_key==None:
        print("Effect size of "+list(main_key.keys())[0]+" "+main_key[list(main_key.keys())[0]][1]+" - "+ main_key[list(main_key.keys())[0]][0])
        h0_idx = seli(df,{list(main_key.keys())[0]:main_key[list(main_key.keys())[0]][0]})
        h1_idx = seli(df,{list(main_key.keys())[0]:main_key[list(main_key.keys())[0]][1]})
    else:
        print("Effect size of "+list(main_key.keys())[0]+" "+main_key[list(main_key.keys())[0]]+" contrasting "+list(sub_key.keys())[0]+" "+sub_key[list(sub_key.keys())[0]][1]+" - "+ sub_key[list(sub_key.keys())[0]][0])
        h0_idx = intersecti([seli(df,main_key),seli(df,{list(sub_key.keys())[0]:sub_key[list(sub_key.keys())[0]][0]})])
        h1_idx = intersecti([seli(df,main_key),seli(df,{list(sub_key.keys())[0]:sub_key[list(sub_key.keys())[0]][1]})])
        
    h0_dat = az_isel_scaled_value(idata,h0_idx).posterior
    h1_dat = az_isel_scaled_value(idata,h1_idx).posterior
    
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
# seli
# -----------------------------------------------------------------------------
# DESCRIPTION
def seli(df,d):
    """
    df ... dataframe
    d ...  dict
    """
    if type(d) is dict:
        s = []
        for key in d:
            s.append( (df[key].reset_index(drop=True)[(df[key].astype('str')==str(d[key])).values]).index.values )
            sel = intersecti(s)
    else:
        print('d must be a dict!!!')
    return sel
# -----------------------------------------------------------------------------
#%% ---------------------------------------------------------------------------
# intersecti
# -----------------------------------------------------------------------------
# DESCRIPTION
def intersecti(a):
    # all inputs must be lists in lists
    for i,b in enumerate(a):
        if i == 0:
            inters = b
        else:
            inters = list(set(inters)&set(b))
    return inters
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
        #n_vars = len(var_names)
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
                    #sel = (df[key].reset_index(drop=True)[(df[key]==item).values]).index.values
                    sel = seli(df,{key:item})
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
                        #kit = (df[key].reset_index(drop=True)[(df[key]==item).values]).index.values
                        #sus = (df[key].reset_index(drop=True)[(df[subset]==sub).values]).index.values
                        #sel = list(set(kit)&set(sus))
                        sel = intersecti([seli(df,{key:item}),seli(df,{subset:sub})])
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
def az_plot_contrast(idat,df,dim,lvl_h0,lvl_h1,t2,mode="global",dim_type='',subdim_type='',ref_val=0, hdi = [-0.1,0.1],sav_plot_q=True):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    import arviz as az
    """
    update to dat-in version
    """
    if mode == "global":
        h0_dat = (dat.posterior.stack(draws=("chain", "draw"))[dim]).sel({dim+'_dim':lvl_h0})
        h1_dat = (dat.posterior.stack(draws=("chain", "draw"))[dim]).sel({dim+'_dim':lvl_h1})
        es_d0, es_d1 = get_data_for_effect_size(idat,df,{dim:[lvl_h0,lvl_h1]})
    else:
        h0_dat = (idat.posterior.stack(draws=("chain", "draw"))[dim_type+':'+subdim_type]).sel({dim_type+':'+subdim_type+'_dim':dim+', '+lvl_h0})
        h1_dat = (idat.posterior.stack(draws=("chain", "draw"))[dim_type+':'+subdim_type]).sel({dim_type+':'+subdim_type+'_dim':dim+', '+lvl_h1})
        es_d0, es_d1 = get_data_for_effect_size(idat,df,{dim_type:dim},{subdim_type:[lvl_h0,lvl_h1]})
            
    bf_df = bayes_factor(h0_dat,h1_dat)
    effs  = effect_size(es_d0, es_d1)
    
    # Plot
    fig,axs = plt.subplots(1,3,figsize=(12,3))
    sns.kdeplot({lvl_h1+'$_\Delta$':h1_dat,lvl_h0+'$_\Delta$':h0_dat}, fill=True, ax=axs[0])
    sns.despine()
    axs[0].set_title(t2+' BF$_{\mathrm{1,0}}$ = '+str(np.round(bf_df.loc['bf10',:][0],2))+', '+
                        'BF$_{\mathrm{0,1}}$ = '+str(np.round(bf_df.loc['bf01',:][0],2)),size=14)
    axs[0].set_xlabel(r'declined < 0 < improved'
                      '\n'
                      "scaled value",size=14)
    axs[0].set_ylabel('density',size=14)
    
    az.plot_posterior(effs, ref_val=ref_val, rope=(hdi[0],hdi[1]),hdi_prob=0.95, ax=axs[1]);
    axs[1].set_title(t2+' '+lvl_h1+'$_\Delta$ - '+lvl_h0+'$_\Delta$ contrast',size=14)
    axs[1].set_xlabel(r'declined < 0 < improved'
                      '\n'
                      "effect size",size=14)
    
    _,p05,p95,ax_ppbf = prior_prob_BF(h0_dat,h1_dat,fig=fig,ax=axs[2])

    plt.gcf().tight_layout()
    if sav_plot_q:
        plt.savefig('bambi_'+dim+'_'+dim_type+'_'+subdim_type+'.pdf')
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
    if t == 'VAS':
        s = s/10
        unit = ' [cm]'
        t_new = 'health VAS'
    elif t == '6minGT_2min':
        s = s/100
        unit = r' [$\times10^2$ m]'
        t_new = 'walk 2 min'
    elif t == 'TUGZeit':
        unit = ' [s]'
        t_new = 'TUG'
    elif t == '10m':
        unit = ' [s]'
        t_new = 'walk 10 m'
    elif t == 'FSST':
        unit = ' [s]'
        t_new = t 
    elif t == 'velocity':
        unit = r' [km/h]'
        t_new = 'gait speed'
        s=s
    elif t == 'StepL_diff':
        unit = r' [$\times10^1$ cm]'
        t_new = 'step length $\Delta$'
        s=s/10
    elif t == 'StanceT_diff':
        unit = r' [$\times10^{-1}$ s]'
        t_new = 'stance time $\Delta$'
        s=s*10
    elif t == 'StanceT_Aside':
        unit = r' [$\times10^{-2}$ s]'
        t_new = 'stance time AL'
        s=s*10
    elif t == 'StanceT_UAside':
        unit = r' [$\times10^{-2}$ s]'
        t_new = 'stance time UL'
        s=s*10
    elif t == 'StepL_Aside':
        unit = r' [$\times10^1$ cm]'
        t_new = 'step length AL'
        s=s/10
    elif t == 'StepL_UAside':
        unit = r' [$\times10^1$ cm]'
        t_new = 'step length UL'
        s=s/10
    return s,unit,t_new
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#%% ---------------------------------------------------------------------------
# get_StanceT_Diff
# -----------------------------------------------------------------------------
# DESCRIPTION
def get_StanceT_Diff(method = 'StanceT_Diff'):
    import pandas as pd
    import numpy as np
    adjustIDs_dict = \
    dict(zip(pd.read_excel("GAITRite/ID_Zuweisung_012023.xlsx",header=2,nrows=13,sheet_name="keep").convert_dtypes()["Datenbank ID"], 
             pd.read_excel("GAITRite/ID_Zuweisung_012023.xlsx",header=2,nrows=13,sheet_name="keep").convert_dtypes()["Pat ID Studie"]))
    adjustSEQ_dict = \
    dict(zip(pd.read_excel("GAITRite/ID_Zuweisung_012023.xlsx",header=2,nrows=13,sheet_name="keep").convert_dtypes()["Pat ID Studie"],
             pd.read_excel("GAITRite/ID_Zuweisung_012023.xlsx",header=2,nrows=13,sheet_name="keep").convert_dtypes()["sequence__"], 
            ))
    ids_meta = pd.read_excel('Metadaten_Patienten_Studie#1.xlsx',header=1,nrows=13,na_values='-')
    ids_meta['color'] = ['blue','lightblue','orange','lightorange','green','lightgreen','red','lightred','purple','lightpurple','brown','lightbrown','pink']
    ids_meta['Geb.Dat. [dd.mm.yyyy]'] = pd.to_datetime(ids_meta['Geb.Dat. [dd.mm.yyyy]'],format="%d.%m.%Y" )
    ids_meta=ids_meta.rename({'PatID':'ID', 'Geschl.':'sex', 'Geb.Dat. [dd.mm.yyyy]':'birthday', 'Größe [cm]':'height_cm', 'Gewicht [kg]':'weight_kg',
                     'Medik.':'medication', 'Proth.beschreib.':'prosthesis', 'Amput.höhe':'amp_level', 'ein/beids.':'aff_side', 'Hilfsmittel':'ass_dev'},axis=1)
    ids_meta = ids_meta.convert_dtypes()
    usecols = ["ID","Date","D","V","Stance_Time_L","Stance_Time_R",'Step_Len_L','Step_Len_R']
    missing_df = pd.read_excel("GAITRite/Datenbank_Exports_Gaitrite.xlsx",header=0,sheet_name="Missing_Stance_Time",usecols=usecols).convert_dtypes()
    
    usecols = ["Pt_ID","Time","Stance_Time_L","Stance_Time_R",'Step_Len_L','Step_Len_R']
    df = pd.read_excel("GAITRite/Datenbank_Exports_Gaitrite.xlsx",header=0,sheet_name="GAITRITE_GaitRiteExport",usecols=usecols).convert_dtypes()
    df = df[df["Pt_ID"].isin(adjustIDs_dict.keys())].convert_dtypes().reset_index(drop=True)
    df["Pt_ID"]=df["Pt_ID"].astype(int)
    df['Date']=pd.to_datetime(df['Time'].dt.date)
    df['Time']=df['Time'].dt.time
    df = df.replace({"Pt_ID":adjustIDs_dict})
    df = df.rename(columns={"Pt_ID":"ID"})

    # remove rows
    # vier Messungen an dem Tag
    df = df.loc[~((df["Date"]=="2022-01-28")*(df["Time"]==pd.to_datetime('09:20:51').time()))].reset_index(drop=True)
    ###
    df = pd.concat([df,missing_df]).reset_index(drop=True)
    df["Sequence"] = df.set_index('ID').rename(adjustSEQ_dict).reset_index().ID.to_list()
    
    df.V = ['V1','V2','V3']*int(156/3)
    d_seqence = [['D0','D0','D0','D1','D1','D1','D2','D2','D2','D3','D3','D3'],['D2','D2','D2','D3','D3','D3','D0','D0','D0','D1','D1','D1']]
    for ID in df['ID'].unique():
        if df[df['ID']==ID]['Sequence'].unique() == "CT":
            sel = 1
        else:
            sel = 0
        df.loc[df['ID']==ID,"D"] = d_seqence[sel]
    df['StanceT_Diff'] = (df['Stance_Time_L'] - df['Stance_Time_R']).abs()
    df.drop(['Time'],axis=1,inplace=True)
    df = df.merge(ids_meta,on='ID')
    
    list_Stance_Time_Aside  = []
    list_Stance_Time_UAside = []
    list_Step_Length_Aside  = []
    list_Step_Length_UAside = []
    for r in range(0,df.shape[0]):
        if df.loc[r,'aff_side']=='LI':
            list_Stance_Time_Aside.append(df.loc[r,'Stance_Time_L'])
            list_Stance_Time_UAside.append(df.loc[r,'Stance_Time_R'])
            list_Step_Length_Aside.append(df.loc[r,'Step_Len_L'])
            list_Step_Length_UAside.append(df.loc[r,'Step_Len_R'])
        if df.loc[r,'aff_side']=='RE':
            list_Stance_Time_Aside.append(df.loc[r,'Stance_Time_R'])
            list_Stance_Time_UAside.append(df.loc[r,'Stance_Time_L'])
            list_Step_Length_Aside.append(df.loc[r,'Step_Len_R'])
            list_Step_Length_UAside.append(df.loc[r,'Step_Len_L'])
    df['StanceT_Aside'] = list_Stance_Time_Aside
    df['StanceT_UAside'] = list_Stance_Time_UAside
    df['StepL_Aside'] = list_Step_Length_Aside
    df['StepL_UAside'] = list_Step_Length_UAside
    df['StanceT_DiffUAmA']=df['StanceT_UAside']-df['StanceT_Aside']
    df['StepL_DiffUAmA']=df['StepL_UAside']-df['StepL_Aside']
    
    multiIDX = pd.MultiIndex.from_tuples(
     [
         ("D0", method,"V1"),
         ("D0", method,"V2"),
         ("D0", method,"V3"),
         ("D2", method,"V1"),
         ("D2", method,"V2"),
         ("D2", method,"V3"),
         ("D1", method,"V1"),
         ("D1", method,"V2"),
         ("D1", method,"V3"),
         ("D3", method,"V1"),
         ("D3", method,"V2"),
         ("D3", method,"V3"),
         ("sequence", "","")
     ],
     names=["", "", ""]
     )
    df_piv = df.pivot(index="ID", columns=["D","V"], values=[method])
    df_piv["sequence"] = df.pivot(index="ID", columns=["D","V"], values=['Sequence']).Sequence.D0.V1
    df_piv.columns = multiIDX
    df_piv = df_piv.reset_index()
    return df_piv
# -----------------------------------------------------------------------------
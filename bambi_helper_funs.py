#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#Created on Sun Mar 24 12:19:33 2019
#
#@author: haraldpenasso

#%% FUNCTIONS TO LOAD
#%%------------------------------------------------------------------------
# hello
# -----------------------------------------------------------------------------
def hello():
    """
    Just prints out "!!! Hello World !!!"
    """
    print('!!! Hello World !!!')
#--------------------------------------------------------------------------
#%% ---------------------------------------------------------------------------
# f_test
# -----------------------------------------------------------------------------
def f_test(a,b):
    import scipy.stats
    import numpy as np
    """
    Compares the variance of two distributions (frequentist)

    INPUT
    ------
    a ... 1d array-like, first distribution
    b ... 1d array-like, second distribution

    OUTPUT
    ------
    p ... float, p-value. p < threshold indicates a difference in variance
    """
    f = np.var(a,ddof=1)/np.var(b,ddof=1) # calculate between-sample F-value
    nun = a.size-1 # shape parameter a
    dun = b.size-1 # shape parameter b
    p = 1 - scipy.stats.f.cdf(f,nun, dun)
    return p
# -----------------------------------------------------------------------------
#%% ---------------------------------------------------------------------------
# summary_stats
# -----------------------------------------------------------------------------
def summary_stats(bf_df,effs,bf_pri_prob,label,rope = [-0.1,0.1],esml = [0.2,0.5,0.8],hdi_prob=0.95):
    import pandas as pd
    import arviz as az
    """
    specific helper function: merge bayes factor and effect size statistics into a table-like form, shows probability for effect sizes lying inside ROPE and effect size classes

    INPUT
    ------
    (use az_plot_contrast to create these inputs)  
    bf_df       ... bayes_factor output, dataframe with bayes factors, prior, probabilities, and odds
    effs        ... effect_size output, arviz or xarray container with posterior effect size data
    bf_pri_prob ... prior_prob_BF output (p05, p95) output as tuple, prior probability required for rejecting (5%) and accepting (95%) H1
    label       ... sting, column name

    optional
    ------
    rope        ... array 1x2, region of practical equivalence. default [-0.1,0.1]
    esml        ... array 1x3, effectsize small medium large boundaries, default [0.2,0.5,0.8]
    hdi_prob    ... float, the % of probability mass to include in the highest density interval

    OUTPUT
    ------
    a pandas DataFrame with one column
    """
    hdi  = az.hdi(effs, hdi_prob=hdi_prob) # arviz highest density interval
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
    },index=[label]).astype(float).transpose()
# -----------------------------------------------------------------------------
#%% ---------------------------------------------------------------------------
# sort_dict_by_mean
# -----------------------------------------------------------------------------
def sort_dict_by_mean(di):
    import numpy as np
    """
    sorts the keys in a dictionary by their associated value mean

    INPUT
    ------
    di ... dictionary

    OUTPUT
    ------
    sorted dictionary
    """
    return {k: v for k, v in sorted(di.items(), key=lambda item: np.mean(item[1]))}
# -----------------------------------------------------------------------------
#%% ---------------------------------------------------------------------------
# sns_to_hex
# -----------------------------------------------------------------------------
def sns_to_hex(sea_col):
    import numpy as np
    """
    converts sns.color_palette("colorblind")[0] to generally usable hex-code

    INPUT
    ------
    sea_col ... seaborn palette color, e.g., sns.color_palette("colorblind")[0]

    OUTPUT
    ------
    the hex-code of the input
    """
    return '#%02x%02x%02x' % tuple([int(i) for i in (np.array(sea_col)*255).round()])
# -----------------------------------------------------------------------------
#%% ---------------------------------------------------------------------------
# sel_data_from_az
# -----------------------------------------------------------------------------
def sel_data_from_az(idata,df,h0_dict,h1_dict=None, kind = None, mode = None, kde_bandwidth = 0.001, random_seed=1234):
    from scipy.stats import gaussian_kde
    import numpy as np
    """
    specific helper function: selects data related to H0 and H1 from arviz inference data

    INPUT
    ------
    idata         ... arviz inference data, bambi2idata4ppc output
    df            ... pandas dataframe used in bmb.Model
    h0_dict       ... kind = effect_size: dict selecting H0 with arviz inference data coordinate name as key, and the specific coordinate as value, e.g.: {'coordinate name A':'coordinate A','coordinate name B':'coordinate B',...}
    				  kind = null: pass H1 arviz inference data coordinate name and coordinate pairs, e.g., {"test":"FSST","intervention":"C","period":"1"}
    				  kind = bayes_factor: uses mode: takes the same input as az_plot_contrast, e.g., {"dim":"lvl_h0"},{"dim":"lvl_h1"}

    optional
    ------
    h1_dict       ... kind = effect_size: dict selecting H0 with arviz inference data coordinate name as key, and the specific coordinate as value, e.g.: {'coordinate name A':'coordinate A','coordinate name B':'coordinate B',...}
    				  kind = null: None
    				  kind = bayes_factor: uses mode: takes the same input as az_plot_contrast, e.g., {"dim":"lvl_h0"},{"dim":"lvl_h1"}
    kind          ... string, effect_size, null, bayes_factor
    mode          ... string, global, tests, ids
    kde_bandwidth ... float, bandwidth setting for scipy.stats gaussian_kde
    random_seed   ... int, sets the pseudo random generator to a reproducible state

    OUTPUT
    ------
    h0_dat        ... arviz inference data for H0
    h1_dat        ... arviz inference data for H1
    """
    if kind == 'null' and h1_dict==None: 
        # based on H0 dict key-item pairs, get boolian selection index
        idx_bool_da0 =  idx_from_dict(h0_dict,df)
        # get data based on idx_bool_da0 (this is actually H1)
        h1_dat = az_isel_scaled_value(idata,idx_bool_da0)['posterior']
        # get data based on idx_bool_da0 (this one will be zero-centered)
        h0_dat = az_isel_scaled_value(idata,idx_bool_da0)['posterior']

        # center-center h0_dat
        h0_dat['scaled_value_mean'] = h0_dat.scaled_value_mean-h0_dat.scaled_value_mean.mean()
        # resample chain estimated mean
        chn = list() # collect chains
        for c in h0_dat.chain:
            obs = list() # collect observations within each chain
            for i,o in enumerate(h0_dat.scaled_value_obs):
            	# get distribution for simulated observation and chain
                vals = h0_dat['scaled_value_mean'].sel(scaled_value_obs=int(o),chain=int(c)).values
                # fit kde, re-sample with updated random_seed, add to list
                obs.append(gaussian_kde(vals,kde_bandwidth).resample(len(vals),random_seed+i)[0])
            # add re-sampled observations to the chain to update
            chn.append(obs)
        # re-order chain dimensions and update the original
        h0_dat['scaled_value_mean'].values = np.swapaxes(chn,1,2) 
        # resample chain estimated sigma
        chn = list() # collect chains
        for i,c in enumerate(h0_dat.chain):
        	# get all values for the chain
            vals = h0_dat['scaled_value_sigma'].sel(chain=int(c)).values
            # fit kde, re-sample with updated random_seed, add to list
            chn.append(gaussian_kde(vals,kde_bandwidth).resample(len(vals),random_seed+i)[0])
        # update the original
        h0_dat['scaled_value_sigma'].values=chn
        
    if kind == 'effect_size' and h1_dict != None:
    	# based on H0 dict key-item pair(s), get boolian selection index
        idx_bool_da0 =  idx_from_dict(h0_dict,df)
        # based on H1 dict key-item pair(s=, get boolian selection index
        idx_bool_da1 =  idx_from_dict(h1_dict,df)
        # get data (mean, sigma will be used) based on idx_bool_da0
        h0_dat = az_isel_scaled_value(idata,idx_bool_da0)['posterior']
        # get data (mean, sigma will be used) based on idx_bool_da1
        h1_dat = az_isel_scaled_value(idata,idx_bool_da1)['posterior']
        
    if kind == "bayes_factor":
        if mode == "global":
        	# one-dimensional selection from the posterior data
            col_H = list(h0_dict.keys())[0] # get the two identical keys, discard the second
            col_H_lvl_h0 = h0_dict[col_H] # get the "value" of the 1st key, i.e., dimension A
            col_H_lvl_h1 = h1_dict[col_H] # get the "value" of the 2nd key, i.e., dimension B
            h0_dat = (idata.posterior.stack(draws=("chain", "draw"))[col_H]).sel({col_H+'_dim':col_H_lvl_h0}) # combine chains and draws, take a data variable, and subset it by the H0 dimension
            h1_dat = (idata.posterior.stack(draws=("chain", "draw"))[col_H]).sel({col_H+'_dim':col_H_lvl_h1}) # combine chains and draws, take a data variable, and subset it by the H1 dimension 
        elif mode == "ids" or mode == "tests":
        	# two-dimensional selection  from the posterior data
            col_H = list(h0_dict.keys())[0] # get the two keys, take the first
            col_A = list(h0_dict.keys())[1] # get the two keys, take the second
            col_A_lvl =    h0_dict[col_A] # get the "value" of the 2nd key, i.e., the level (e.g., test)
            col_H_lvl_h0 = h0_dict[col_H] # get the "value" of the 1st key, i.e., dimension H0
            col_H_lvl_h1 = h1_dict[col_H] # get the "value" of the 1st key, i.e., dimension H1
            h0_dat = (idata.posterior.stack(draws=("chain", "draw"))[col_A+':'+col_H]).sel({col_A+':'+col_H+'_dim':col_A_lvl+', '+col_H_lvl_h0}) # combine chains and draws, take a data variable, and subset it by the H0 dimension
            h1_dat = (idata.posterior.stack(draws=("chain", "draw"))[col_A+':'+col_H]).sel({col_A+':'+col_H+'_dim':col_A_lvl+', '+col_H_lvl_h1}) # combine chains and draws, take a data variable, and subset it by the H1 dimension
            
    return h0_dat, h1_dat
# -----------------------------------------------------------------------------
#%% ---------------------------------------------------------------------------
# az_effect_size
# -----------------------------------------------------------------------------
def effect_size(h0_dat,h1_dat,sig_diff=0.05):
    """
    effect size of H1 - H0, calculates hedge's g or glass delta if the groups' variance differs

    INPUT
    ------
    h0_dat   ... arviz inference data, output of sel_data_from_az for H0
    h1_dat   ... arviz inference data, output of sel_data_from_az for H0

    optional
    ------
    sig_diff ... float, frequentist p-value threshold for F-test testing variance difference, default 0.05

    OUTPUT
    ------
    arviz inference data of effect size for difference of H1 - H0
    """
    n_h0 = h0_dat['scaled_value_sigma'].size # sample size of H0
    n_h1 = h1_dat['scaled_value_sigma'].size # sample size of H1
    n = n_h1 + n_h0 # total sample size
    
    if f_test(h0_dat['scaled_value_sigma'],h1_dat['scaled_value_sigma'])<sig_diff:
    	# Glass delta uses only the standard deviation of H0 data
        print("Computing corrected Glass's delta")
        sd_pooled = h0_dat['scaled_value_sigma']
    else:
    	# Hedges g uses the pooled standard deviation of H0 and H1 data
        print("Computing corrected Hedge's g")
        sd_pooled = ( ( (n_h0-1)*h0_dat['scaled_value_sigma']**2 + (n_h1-1)*h1_dat['scaled_value_sigma']**2) / (n_h0+n_h1-2) )**(1/2)
    # calculate the difference of the means
    diff_of_means = h1_dat['scaled_value_mean'].mean(axis=2).values-h0_dat['scaled_value_mean'].mean(axis=2).values
    # caclulate cohen's d and apply Hedges correction for small sample sizes (the correction will be one with bayesian posterior data)
    return (diff_of_means/sd_pooled) * ((n-3)/(n-2.25))*((n-2)/n)**(1/2)
# -----------------------------------------------------------------------------

#%% ---------------------------------------------------------------------------
# bambi2idata4ppc
# -----------------------------------------------------------------------------
def bambi2idata4ppc(model,result,draws=6000,random_seed = 1234):
    """
    add predicted posterior distribution of the mean, posterior predictive, and prior predictive distribution

    INPUT
    ------
    model       ... BAMBI model
    result      ... fitted BAMBI model results

    optional
    ------
    draws       ... int, number of prior predictive samples simulated, default: 6000
    random_seed ... int, pseudo random generator setting for reproducibility, default: 1234

    OUTPUT
    ------
    idata       ... arviz inference data with predictions
    """
    model.predict(result, inplace=True, kind="mean") # draws from the posterior distribution of the mean (mean and sigma)
    idata = model.predict(result, inplace=False, kind="pps") # draws from the posterior predictive distribution
    idata.extend( model.prior_predictive(draws=draws,random_seed=random_seed) ) # prior predictive
    return idata
# -----------------------------------------------------------------------------
#%% ---------------------------------------------------------------------------
# az_isel_scaled_value
# -----------------------------------------------------------------------------
def az_isel_scaled_value(idata,sel):
    """
    subset arviz inference data using an index array

    INPUT
    ------
    idata ... arviz inference data
    sel   ... array, index

    OUTPUT
    ------
    idata ... subset of arviz inference data
    """
    # use isel method on XXX_obs, XXX_dim_0, ..., where XXX is the dependent variable, e.g., scaled_value
    idata = idata.isel(scaled_value_obs=sel).isel(scaled_value_dim_0=sel)
    return idata
# -----------------------------------------------------------------------------
#%% ---------------------------------------------------------------------------
# ppc_bambi
# -----------------------------------------------------------------------------
# could remove subset option using new intersecti fun 
def ppc_bambi(idata,df,var_names=[],subset=[],check_names = ['prior','posterior'],n_pp_samp=[],colors=['#5AA36A','k','orange'], kind='cumulative',random_seed = 1234):
    import matplotlib.pyplot as plt
    import arviz as az
    """
    specific plot organization function, plot prior and posterior predictive checks next to each other

    INPUT
    ------
    idata      ... arviz inference data, e.g., the output of bambi2idata4ppc
    df         ... pandas dataframe, the df used when fitting the BAMBI model

    optional
    ------
    var_names   ... list of strings, i.e., a fixed effect with all levels plotted as a row
    subset      ... string, i.e., additional condition with levels plotted side-by-side
    check_names ... list of strings, define the label of the prior and posterior data
    n_pp_samp   ... int, number of samples to plot; passing [] will plot all
    colors      ... list of strings, define 3 colors: prior predictive, observed, prior predictive mean
    kind        ... string, passed to az.plot_ppc either kde or cumulative
    random_seed ... int, pseudo random generator setting for reproducibility, default: 1234

    OUTPUT
    ------
    fig ... plt figure handle
    axs ... plt axis handle
    """
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
def az_plot_contrast(idata,df,h0_dict,h1_dict,t2,mode=None,ref_val=0, rope = [-0.1,0.1],sav_plot_q=True):    
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    import arviz as az
    """
    plot H0, H1 distributions, their difference, and prior probability required for accepting a difference between the distributions

    INPUT
    ------
    idata      ... arviz inference data, e.g., the output of bambi2idata4ppc
    df         ... pandas dataframe, the df used when fitting the BAMBI model
    h0_dict    ... dict, key-value pairs for selecting H0 data, e.g., {'intervention':lvl_h0,'test':dim}
    h1_dict    ... dict, key-value pairs for selecting H1 data, e.g., {'intervention':lvl_h1,'test':dim}
    t2         ... string, title or name of the parameter shown on the plot

    optional
    ------
    mode       ... string, global, tests, ids
    ref_val    ... float, reference value passed to az.plot_posterior
    rope       ... iterable length 2, lower and upper boundary of region of practical equivalence
    sav_plot_q ... bool, set to False to not save the plot as PDF

    OUTPUT
    ------
    bf_df      ... dataframe, Bayes factors output from bayes_factor()
    effs       ... xarray.DataArray, effect size distribution, the output of effect_size()
    (p05,p95)  ... tuple with floats, the prior probability s.t. posterior belief < 0.05 and < 0.95 for accepting a difference in the distributions
    """
    # get data
    bf_d0, bf_d1 = sel_data_from_az(idata,df,h0_dict,h1_dict,kind="bayes_factor",mode=mode) # for Bayes factor
    es_d0, es_d1 = sel_data_from_az(idata,df,h0_dict,h1_dict,kind="effect_size") # for effect size
    
    # calculate
    bf_df = bayes_factor(bf_d0,bf_d1) # Bayes factors
    effs  = effect_size(es_d0, es_d1) # effect size

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
    
    az.plot_posterior(effs, ref_val=ref_val, rope=(rope[0],rope[1]),hdi_prob=0.95, ax=axs[1]);
    axs[1].set_title(t2+' '+col_H_lvl_h1+'$_\Delta$ - '+col_H_lvl_h0+'$_\Delta$ contrast',size=14)
    axs[1].set_xlabel(r'declined < 0 < improved'
                      '\n'
                      "effect size",size=14)
    
    # calculate prior probability for accepting a difference between distributions, get its plot/axis
    _,p05,p95,ax_ppbf = prior_prob_BF(bf_d0,bf_d1,fig=fig,ax=axs[2])
    plt.gcf().tight_layout()
    
    # option to save plot
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
    calculate Bayes factor for difference between two distributions
    
    INPUT
    ------
    da0           ... arviz inference data, posterior distribution for H0
    da1           ... arviz inference data, posterior distribution for H1

    optional
    ------
    prior_sim     ... float, prior probability favouring similarity between distributions (prior_sim + prior_diff != 1)
    prior_diff    ... float, prior probability favouring difference between distributions (prior_sim + prior_diff != 1)
    kde_bandwidth ... float, bandwidth setting for scipy.stats gaussian_kde

    OUTPUT
    ------
    dataframe with one column of prior probability, probabilities, odds, and Bayes factors
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
    
    # Organize the output 
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
                
    optional
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
def get_prior_probs(vals,prior_probs):
    from scipy.interpolate import CubicSpline
    """
    function used by prior_prob_BF to estimate 0.05 and 0.95 probability thresholds
    
    INPUT
    ------
    vals        ... list of floats, y-axis, each the first cell of the df output of bayes_factor(), i.e., the posterior probability for accepting a difference between distributions
    prior_probs ... list of floats, x-axis, i.e., the prior probability for accepting a difference between distributions
    
    OUTPUT
    ------
    lims        ... list, length 2, prior probability s.t. posterior < 0.05 and < 0.95, resp.
    """
    if sum(vals)+1==float(len(vals)):
        lims = [0.,0.] # if all posterior probabilities equal 1, set prior probability to 0 at both thresholds
    else:
        cs = CubicSpline(vals, prior_probs) # fit data with cubic splines
        lims = [cs(0.05),cs(0.95)] # evaluate fit at threshold level
    return lims
# -----------------------------------------------------------------------------

#%% ---------------------------------------------------------------------------
# prior_prob_BF
# -----------------------------------------------------------------------------
def prior_prob_BF(da0, da1, fig, ax, stepsize = 0.001,figsize=(4.5,4.5)):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    """
    calculate prior probabilities for accepting difference between distributions
    
    INPUT
    ------
    da0      ... arviz inference data, posterior distribution for H0
    da1      ... arviz inference data, posterior distribution for H1
    fig      ... figure handle passed by az_plot_contrast()
    ax       ... axis handle passed by az_plot_contrast()
    
    optional
    ------
    stepsize ... float, step-size of the x-axis, i.e., prior probabilities, default 0.001
    figsize  ... tuple, length 2, sets figsize parameter, default (4.5,4.5)
    
    OUTPUT
    ------
    bf       ... dataframe, with Bayes factors, the output of bayes_factor()
    p05      ... float, prior probability s.t. posterior < 0.05
    p95      ... float, prior probability s.t. posterior < 0.95
    ax       ... axis handle
    """
    p_pris = np.arange(10**(-16),1+stepsize,stepsize) # set array of prior probability values, cant begin at zero becaus we'd have a 1/0 situation
    vals = [0] # initial posterior probability value at min(p_pris) set to zero
    for p_pri in p_pris:
    	# calculate posterior probability for each prior probability
        vals.append(bayes_factor(da0, da1, prior_sim=1-p_pri, prior_diff=p_pri).loc['pH1gD',:].to_list()[0])
    p_pris = np.insert(p_pris,0,0) # add a zero to the beginning of p_pris
    bf = bayes_factor(da0, da1, prior_sim=0.5, prior_diff=0.5).loc['bf10',:].to_list()[0] # calculate the regular Bayes factor for the plot
    # some corrections the cubic spline fit returned values greater or smaller than 0 and 1
    p05,p95 = get_prior_probs(vals,p_pris)
    if (p95 > 1):
        p95 = 1.0
    elif (p95 < 0):
        p95 = 0.0
    if (p05 > 1):
        p05 = 1.0
    elif (p05 < 0):
        p05 = 0.0
    # Plot (some local figure save functionality not yet removed)
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
#%% ---------------------------------------------------------------------------
# scaleback
# -----------------------------------------------------------------------------
def scaleback(t,df):
    """
    specific helper function: get scale value, unit, updated variable name for plots
    
    INPUT
    ------
    t     ... string, variable name
    df    ... dataframe with the data used in the BAMBI model
    
    OUTPUT
    ------
    s     ... float, scale value
    unit  ... string, unit
    t_new ... string, updated variable name
    """
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
    elif t == 'StepLengthDiff':
        unit = r' [$\times10^2$ cm]'
        t_new = 'step length diff'
        s=s/100
    return s,unit,t_new
# -----------------------------------------------------------------------------
#%% ---------------------------------------------------------------------------
# idx_from_dict
# -----------------------------------------------------------------------------
def idx_from_dict(d,df):
    """
    create boolian selection array from dict and dataframe
    
    INPUT
    ------
    d        ... dict, each key corresponds to a df column and each value to a cell entry of that column
    df       ... dataframe, the boolian index based on the dict

    OUTPUT
    ------
    idx_bool ... combined matches as boolian array
    """
    for i,k in enumerate(d):
        if i == 0:
            idx_bool = df[k].astype('str')==str(d[k])
        else:
            idx_bool = idx_bool * (df[k].astype('str')==str(d[k]))
    return idx_bool
# -----------------------------------------------------------------------------
#%% ---------------------------------------------------------------------------
# closest_value
# -----------------------------------------------------------------------------
# DESCRIPTION
def closest_value(input_data, input_value):
    """
    find closest index and value to input_value in input_data
    returns (index, closest_values)
    https://www.entechin.com/find-nearest-value-list-python"
    """
    import numpy as np
    difference = lambda input_data : abs(input_data - input_value)
    res = min(input_data, key=difference)
    return (np.where(input_data==res),res)
# -----------------------------------------------------------------------------
#%% ---------------------------------------------------------------------------
# intersect_two_dist
# -----------------------------------------------------------------------------
# DESCRIPTION
def intersect_two_dist(adat,bdat,stepsize=1000,pltQ=False):
    """
    find the intersection point of two data distributions
    adat ... data of the first distribution
    bdat ... data of the second distribution
    stepsize ... for evaluating x between kde peaks
    """
    import numpy as np
    from scipy.optimize import minimize
    from scipy import stats
    # determine which one is left/right orientation
    if np.max([adat.mean(),bdat.mean()]) == adat.mean():
        mn_right = adat.mean()
        dat_right= adat
        right_pdf=stats.gaussian_kde(adat)
        mn_left = bdat.mean()
        dat_left = bdat
        left_pdf=stats.gaussian_kde(bdat)
    else:
        mn_left = adat.mean()
        dat_left = adat
        left_pdf=stats.gaussian_kde(adat)
        mn_right = bdat.mean()
        dat_right= bdat
        right_pdf=stats.gaussian_kde(bdat)

    # get some x values between peaks for searching the intersection point 
    x_between = np.arange(mn_left,mn_right,stepsize/adat.size)
    
    # deviding the two parts of the distributions is close to 1 at the point, fit pdfs and find an estimate
    closest_to_one = closest_value(left_pdf(x_between)/right_pdf(x_between), 1)
    
    # optimize the estimate using closest_to_one as initial guess and the peaks as bounds
    # bnds = ((mn_left, mn_right), (mn_left, mn_right)) those bounds do not work for the priors
    intersect = minimize(lambda x: np.abs(1-left_pdf(x)/right_pdf(x))[0], x_between[closest_to_one[0]], method='Nelder-Mead', tol=1e-6)#, bounds=bnds
    
    # probability that the distributions are equal
    p_eq = left_pdf.integrate_box_1d( intersect.x[0],np.inf) + right_pdf.integrate_box_1d(-np.inf,intersect.x[0])
    if p_eq > 1:
        print("!!! WARNING: The probability of equality was calculated > "+str(p_eq)+"; setting it to 1")
        p_eq = 1
        
    if pltQ == True:
        import matplotlib.pyplot as plt
        import seaborn as sns
        print('The intersection point is at '+str( np.round(intersect.x[0],3))+' with y error '+str((1-left_pdf(intersect.x[0])/right_pdf(intersect.x[0]))[0]))
        sns.kdeplot({'left':dat_left,'right':dat_right},common_norm=False)
        plt.plot([intersect.x[0],intersect.x[0]],[0,left_pdf(intersect.x)[0]],'o-',alpha=0.5)
        plt.plot([intersect.x[0],intersect.x[0]],[0,right_pdf(intersect.x)[0]],'o-',alpha=0.5)
        plt.title('The probability that the distributions are equal is '+str(np.round(p_eq,3)))
        plt.show()
    return p_eq
# -----------------------------------------------------------------------------
"""
#%% ---------------------------------------------------------------------------
# az_bf
# -----------------------------------------------------------------------------
# DESCRIPTION
def az_bf(idata, key, lvl_h0, lvl_h1, ref_val = "equal", prio_odds_set = 1, pltQ = False):
    " ""
    set ref_val to "equal" (default) for posterior testing H0 that lvl_h0 == lvl_h1 (H1 that lvl_h0 != lvl_h1)
    or
    set ref_val to a number to for testing hypothesis at that point
    
    use prio_odds_set = 1 (default) to provide H0 and H1 with the same probabilty,
    or
    use None to get the prior belief from the prior distribution
    " ""
    # prio_prob_set list of two numers h0 h1 resp
    # https://en.wikipedia.org/wiki/Bayes_factor
    # https://bookdown.org/kevin_davisross/bayesian-reasoning-and-methods/bayes-factor.html
    # https://doi.org/10.3758/s13423-020-01798-5
    # Bayes Factor approximated as the Savage-Dickey density ratio.
    # https://github.com/GStechschulte/arviz/commit/e41f0d0f78e969712d26e835ea21a916c0caaefd
    # import pandas as pd
    # from scipy import stats
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    h1_dat_post = (idata.posterior.stack(draws=("chain", "draw"))[key]).sel({key+'_dim':lvl_h1})
    h0_dat_post = (idata.posterior.stack(draws=("chain", "draw"))[key]).sel({key+'_dim':lvl_h0})

    if (type(ref_val) == float) or (type(ref_val) == int):
        h1_pdf_post = stats.gaussian_kde(h1_dat_post) # get kde for h1 samples in posterior
        h1_prob_post_at_ref = h1_pdf_post(ref_val)[0] # get h1 probability in posterior at reference value
        h0_pdf_post = stats.gaussian_kde(h0_dat_post)
        h0_prob_post_at_ref = h0_pdf_post(ref_val)[0]
        if prio_odds_set != None:
            # set prior odds for senitivity analzsis
            h0_prob_prio_at_ref = 1/prio_odds_set
            h1_prob_prio_at_ref = 1
        else:
            h1_dat_prio = (idata.prior.stack(draws=("chain", "draw"))[key]).sel({key+'_dim':lvl_h1})
            h0_dat_prio = (idata.prior.stack(draws=("chain", "draw"))[key]).sel({key+'_dim':lvl_h0})
            h1_pdf_prio = stats.gaussian_kde(h1_dat_prio) # get kde for h1 samples in prior
            h1_prob_prio_at_ref = h1_pdf_prio(ref_val)[0] # get h1 probability in prior at reference value
            h0_pdf_prio = stats.gaussian_kde(h0_dat_prio)
            h0_prob_prio_at_ref = h0_pdf_prio(ref_val)[0]
        
    elif ref_val == 'equal':
        h0_prob_post_at_ref = intersect_two_dist(h0_dat_post,h1_dat_post)
        h1_prob_post_at_ref = 1 - h0_prob_post_at_ref
        if prio_odds_set != None:
            # set prior odds for senitivity analysis
            h0_prob_prio_at_ref = 1/prio_odds_set
            h1_prob_prio_at_ref = 1
        else:
            h1_dat_prio = (idata.prior.stack(draws=("chain", "draw"))[key]).sel({key+'_dim':lvl_h1})
            h0_dat_prio = (idata.prior.stack(draws=("chain", "draw"))[key]).sel({key+'_dim':lvl_h0})
            h0_prob_prio_at_ref = intersect_two_dist(h0_dat_prio,h1_dat_prio)
            h1_prob_prio_at_ref = 1 - h0_prob_prio_at_ref     
    else:
        print('ref_val was not correctly specified!')
    # Odds
    odds_prio = h1_prob_prio_at_ref/h0_prob_prio_at_ref
    odds_post = h1_prob_post_at_ref/h0_prob_post_at_ref
    # Bayesian Factors
    bf10 = odds_post/odds_prio
    bf01 = 1/bf10
    # Plots
    if pltQ == True:
        fig, axs = plt.subplots(1,2,figsize=(12,3))
        if prio_odds_set == None:
            sns.kdeplot({lvl_h1:h1_dat_prio,lvl_h0:h0_dat_prio},common_norm=False,ax=axs[0])
        sns.kdeplot({lvl_h1:h1_dat_post,lvl_h0:h0_dat_post},common_norm=False,ax=axs[1])
        axs[0].plot([ref_val,ref_val],[h1_prob_prio_at_ref,h0_prob_prio_at_ref],'k-o',alpha = 0.7)
        axs[1].plot([ref_val,ref_val],[h1_prob_post_at_ref,h0_prob_post_at_ref],'k-o',alpha = 0.7)
        axs[0].plot([ref_val,ref_val],[0,np.min([h1_prob_prio_at_ref,h0_prob_prio_at_ref])],'k:')
        axs[1].plot([ref_val,ref_val],[0,np.min([h1_prob_post_at_ref,h0_prob_post_at_ref])],'k:')
        axs[0].set_title('Prior')
        axs[1].set_title('Posterior')
        plt.show()
    if (type(ref_val) == float) or (type(ref_val) == int):
        print('BF10 ('+lvl_h1+'=='+str(ref_val)+')'+' = '+str(np.round(bf10,3))+'\nBF01 ('+lvl_h0+'=='+str(ref_val)+') = '+str(np.round(bf01,3)))
    d = {'pH1':h1_prob_prio_at_ref,
         'pH0':h0_prob_prio_at_ref,
         'pH1gD':h1_prob_post_at_ref,
         'pH0gD':h0_prob_post_at_ref,
         'prior_odds': odds_prio,
         'posterior_odds': odds_post,
         'bf10': bf10,
         'bf01': bf01 }
    return pd.DataFrame(d,index=['Test_on_'+str(ref_val)]).astype(float).transpose()
"""

def az_bf(idata, key, lvl_h0, lvl_h1, prior_sim=0.5, prior_diff=0.5,kde_bandwidth = 0.001):
    import xarray as xr
    import numpy as np
    from scipy.stats import gaussian_kde #norm
    import pandas as pd
    """
    self data selecting version
    """
    # selected xarray.DataArray(s)
    da1 = (idata.posterior.stack(draws=("chain", "draw"))[key]).sel({key+'_dim':lvl_h1})
    da2 = (idata.posterior.stack(draws=("chain", "draw"))[key]).sel({key+'_dim':lvl_h0})
    
    # Compute the mean and standard deviation of the two input DataArrays
    mean1 = da1.mean().values
    #std1 = da1.std().values
    mean2 = da2.mean().values
    #std2 = da2.std().values
    
    # Compute the likelihood for the similarity hypothesis
    # sd and mean method
    #odds_sim = (norm(mean1, std1).pdf(mean2) * norm(mean2, std2).pdf(mean1))
    # KDE methods
    # sklearn may provides more kernels
        #from sklearn.neighbors import KernelDensity as kde
        #np.exp(kde(kernel='epanechnikov',bandwidth=0.005).fit([[i] for i in da1.values]).score_samples([[mean2]]))
    odds_sim = gaussian_kde(da1,kde_bandwidth).pdf(mean2) * gaussian_kde(da2,kde_bandwidth).pdf(mean1)
    
    # Compute the likelihood for the difference hypothesis
    #odds_diff = (norm(mean1, std1).pdf(mean1) * norm(mean2, std2).pdf(mean2))
    odds_diff = gaussian_kde(da1,kde_bandwidth).pdf(mean1) * gaussian_kde(da2,kde_bandwidth).pdf(mean2)
    
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

def az_bf(idata, key, lvl_h0, lvl_h1, prior_sim=0.5, prior_diff=0.5,kde_bandwidth = 0.001):
    import xarray as xr
    import numpy as np
    from scipy.stats import gaussian_kde #norm
    import pandas as pd
    """
    Version that may compare to prior
    """
    # selected xarray.DataArray(s)
    da1 = (idata.posterior.stack(draws=("chain", "draw"))[key]).sel({key+'_dim':lvl_h1})
    if lvl_h0.split(", ")[1] == 'prior' or lvl_h0=='prior':
        da2 = (idata.prior_adj2post.stack(draws=("chain", "draw"))[key]).sel({key+'_dim':lvl_h1})
    else:
        da2 = (idata.posterior.stack(draws=("chain", "draw"))[key]).sel({key+'_dim':lvl_h0})
    
    # Compute the mean and standard deviation of the two input DataArrays
    mean1 = da1.mean().values
    #std1 = da1.std().values
    mean2 = da2.mean().values
    #std2 = da2.std().values
    
    # Compute the likelihood for the similarity hypothesis
    # sd and mean method
    #odds_sim = (norm(mean1, std1).pdf(mean2) * norm(mean2, std2).pdf(mean1))
    # KDE methods
    # sklearn may provides more kernels
        #from sklearn.neighbors import KernelDensity as kde
        #np.exp(kde(kernel='epanechnikov',bandwidth=0.005).fit([[i] for i in da1.values]).score_samples([[mean2]]))
    odds_sim = gaussian_kde(da1,kde_bandwidth).pdf(mean2) * gaussian_kde(da2,kde_bandwidth).pdf(mean1)
    
    # Compute the likelihood for the difference hypothesis
    #odds_diff = (norm(mean1, std1).pdf(mean1) * norm(mean2, std2).pdf(mean2))
    odds_diff = gaussian_kde(da1,kde_bandwidth).pdf(mean1) * gaussian_kde(da2,kde_bandwidth).pdf(mean2)
    
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

def az_effect_size(idata,df,main_key,sub_key=None,sig_diff=0.05):
    """
    Version that may compare to prior
    """
    # https://www.statisticshowto.com/hedges-g/
    # (!) https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/hedgeg.htm
    # https://rowannicholls.github.io/python/statistics/effect_size.html#hedgess-g
    # https://www.socscistatistics.com/effectsize/default3.aspx
    # https://rdrr.io/cran/rstatix/src/R/cohens_d.R
    # main_key ... dict with main_key value pair of test_col:test or main_kez:[level_control, level_treat] without a subkey
    # sub_key ... dict with main_key value pair of subset_col:[level_control, level_treat]
    
    if sub_key==None:
        print("Effect size of "+list(main_key.keys())[0]+" "+main_key[list(main_key.keys())[0]][1]+" - "+ main_key[list(main_key.keys())[0]][0])
        h0_idx = seli(df,{list(main_key.keys())[0]:main_key[list(main_key.keys())[0]][0]})
        h1_idx = seli(df,{list(main_key.keys())[0]:main_key[list(main_key.keys())[0]][1]})
    else:
        print("Effect size of "+list(main_key.keys())[0]+" "+main_key[list(main_key.keys())[0]]+" contrasting "+list(sub_key.keys())[0]+" "+sub_key[list(sub_key.keys())[0]][1]+" - "+ sub_key[list(sub_key.keys())[0]][0])
        h0_idx = intersecti([seli(df,main_key),seli(df,{list(sub_key.keys())[0]:sub_key[list(sub_key.keys())[0]][0]})])
        h1_idx = intersecti([seli(df,main_key),seli(df,{list(sub_key.keys())[0]:sub_key[list(sub_key.keys())[0]][1]})])
    
    
    if sub_key[list(sub_key.keys())[0]][0]=='prior':
        h0_dat = az_isel_scaled_value(idata,h1_idx).prior_adj2post
    else:
        h0_dat = az_isel_scaled_value(idata,h0_idx).posterior
    h1_dat = az_isel_scaled_value(idata,h1_idx).posterior
    n_h0 = h0_dat['scaled_value_sigma'].size
    n_h1 = h1_dat['scaled_value_sigma'].size
    n = n_h1 + n_h0
    if f_test(h0_dat['scaled_value_sigma'],h1_dat['scaled_value_sigma'])<sig_diff:
        if sub_key==None:
            print("Computing corrected Glass's delta, i.e., only SD of "+main_key[list(main_key.keys())[0]][0]+" (should be controls!)")
        else:
            print("Computing corrected Glass's delta, i.e., only SD of "+sub_key[list(sub_key.keys())[0]][0]+" (should be controls!)")
        sd_pooled = h0_dat['scaled_value_sigma']
    else:
        print("Computing corrected Hedge's g")
        sd_pooled = ( ( (n_h0-1)*h0_dat['scaled_value_sigma']**2 + (n_h1-1)*h1_dat['scaled_value_sigma']**2) / (n_h0+n_h1-2) )**(1/2)
    diff_of_means = h1_dat['scaled_value_mean'].mean(axis=2).values-h0_dat['scaled_value_mean'].mean(axis=2).values
    return (diff_of_means/sd_pooled) * ((n-3)/(n-2.25))*((n-2)/n)**(1/2)
    
def az_plot_contrast(idat,df,dim,lvl_h0,lvl_h1,mode,t2,dim_type='',subdim_type='',ref_val=0, hdi = [-0.1,0.1],sav_plot_q=True):
    """
    Version that may compare to prior
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    import arviz as az
    
    if lvl_h0 == 'prior':
        lvl_h0_dat = (idat.prior_adj2post.stack(draws=("chain", "draw"))[dim_type+':'+subdim_type]).sel({dim_type+':'+subdim_type+'_dim':dim+', '+lvl_h1})
    else:
        lvl_h0_dat = (idat.posterior.stack(draws=("chain", "draw"))[dim_type+':'+subdim_type]).sel({dim_type+':'+subdim_type+'_dim':dim+', '+lvl_h0})
    lvl_h1_dat = (idat.posterior.stack(draws=("chain", "draw"))[dim_type+':'+subdim_type]).sel({dim_type+':'+subdim_type+'_dim':dim+', '+lvl_h1})
    if mode == 'global':
        bf_df = az_bf(idat,key=dim,lvl_h0=lvl_h0,lvl_h1=lvl_h1)
        effs = az_effect_size(idat,df,{dim:[lvl_h0,lvl_h1]})
    elif mode == 'tests':
        bf_df = az_bf(idat,key=dim_type+':'+subdim_type,lvl_h0=dim+', '+lvl_h0,lvl_h1=dim+', '+lvl_h1)
        effs = az_effect_size(idat,df,{dim_type:dim},{subdim_type:[lvl_h0,lvl_h1]})
    # Plot
    fig,axs = plt.subplots(1,3,figsize=(12,3))
    sns.kdeplot({lvl_h1+'$_\Delta$':lvl_h1_dat,lvl_h0+'$_\Delta$':lvl_h0_dat}, fill=True, ax=axs[0])
    sns.despine();
    axs[0].set_title(t2+' BF$_{\mathrm{1,0}}$ = '+str(np.round(bf_df.loc['bf10',:][0],2))+', '+
                        'BF$_{\mathrm{0,1}}$ = '+str(np.round(bf_df.loc['bf01',:][0],2)),size=14)
    axs[0].set_xlabel(r'declined < 0 < improved'
                      '\n'
                      "scaled value",size=14)
    axs[0].set_ylabel('density',size=14)
    
    az.plot_posterior(effs, ref_val=ref_val, rope=(hdi[0],hdi[1]),
                  hdi_prob=0.95, ax=axs[1]);
    axs[1].set_title(t2+' '+lvl_h1+'$_\Delta$ - '+lvl_h0+'$_\Delta$ contrast',size=14)

    axs[1].set_xlabel(r'declined < 0 < improved'
                      '\n'
                      "effect size",size=14)
    # ---
    if mode == 'global':
        _,p05,p95,ax_ppbf = prior_prob_BF(idat, dim, dim+', '+lvl_h0, dim+', '+lvl_h1,fig=fig,ax=axs[2])
    elif mode == 'tests':
        _,p05,p95,ax_ppbf = prior_prob_BF(idat, dim_type+':'+subdim_type, dim+', '+lvl_h0, dim+', '+lvl_h1,fig=fig,ax=axs[2])

    # ---
    plt.gcf().tight_layout()
    if sav_plot_q:
        plt.savefig('bambi_'+dim+'_'+dim_type+'_'+subdim_type+'.pdf')
    plt.show()
    return bf_df, effs, (p05,p95)

def az_effect_size(idata,df,main_key,sub_key=None,sig_diff=0.05):
    # https://www.statisticshowto.com/hedges-g/
    # (!) https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/hedgeg.htm
    # https://rowannicholls.github.io/python/statistics/effect_size.html#hedgess-g
    # https://www.socscistatistics.com/effectsize/default3.aspx
    # https://rdrr.io/cran/rstatix/src/R/cohens_d.R
    # main_key ... dict with main_key value pair of test_col:test or main_kez:[level_control, level_treat] without a subkey
    # sub_key ... dict with main_key value pair of subset_col:[level_control, level_treat]
    """
    self- data selecting version
    """
    if sub_key==None:
        print("Effect size of "+list(main_key.keys())[0]+" "+main_key[list(main_key.keys())[0]][1]+" - "+ main_key[list(main_key.keys())[0]][0])
        h0_idx = seli(df,{list(main_key.keys())[0]:main_key[list(main_key.keys())[0]][0]})
        h1_idx = seli(df,{list(main_key.keys())[0]:main_key[list(main_key.keys())[0]][1]})
    else:
        print("Effect size of "+list(main_key.keys())[0]+" "+main_key[list(main_key.keys())[0]]+" contrasting "+list(sub_key.keys())[0]+" "+sub_key[list(sub_key.keys())[0]][1]+" - "+ sub_key[list(sub_key.keys())[0]][0])
        if sub_key[list(sub_key.keys())[0]][0]!='prior':
            h0_idx = intersecti([seli(df,main_key),seli(df,{list(sub_key.keys())[0]:sub_key[list(sub_key.keys())[0]][0]})])
        h1_idx = intersecti([seli(df,main_key),seli(df,{list(sub_key.keys())[0]:sub_key[list(sub_key.keys())[0]][1]})])

    h0_dat = az_isel_scaled_value(idata,h0_idx).posterior
    h1_dat = az_isel_scaled_value(idata,h1_idx).posterior
    n_h0 = h0_dat['scaled_value_sigma'].size
    n_h1 = h1_dat['scaled_value_sigma'].size
    n = n_h1 + n_h0
    if f_test(h0_dat['scaled_value_sigma'],h1_dat['scaled_value_sigma'])<sig_diff:
        if sub_key==None:
            print("Computing corrected Glass's delta, i.e., only SD of "+main_key[list(main_key.keys())[0]][0]+" (should be controls!)")
        else:
            print("Computing corrected Glass's delta, i.e., only SD of "+sub_key[list(sub_key.keys())[0]][0]+" (should be controls!)")
        sd_pooled = h0_dat['scaled_value_sigma']
    else:
        print("Computing corrected Hedge's g")
        sd_pooled = ( ( (n_h0-1)*h0_dat['scaled_value_sigma']**2 + (n_h1-1)*h1_dat['scaled_value_sigma']**2) / (n_h0+n_h1-2) )**(1/2)
    diff_of_means = h1_dat['scaled_value_mean'].mean(axis=2).values-h0_dat['scaled_value_mean'].mean(axis=2).values
    return (diff_of_means/sd_pooled) * ((n-3)/(n-2.25))*((n-2)/n)**(1/2)


def xr_resample_data(azInfDat,dim1,dim2,dim0='prior',dim3='data',random_seed = 1234,kde_bandwidth = 0.001):
    """
    elif len(azInfDat[dim0].to_dict()[dim1][dim2]['dims']) == 4:
        if np.shape(azInfDat[dim0].to_dict()[dim1][dim2][dim3])[3] == 1:
            return xr.DataArray.from_dict({
                "dims":  azInfDat[dim0].to_dict()[dim1][dim2]['dims'],
                "attrs": {},
                "coords":azInfDat.posterior[dim2].to_dict()['coords'],
                "name":  dim2,
                "data":  np.expand_dims(np.array([np.apply_along_axis(reshape_fun,0,
                               np.matrix(np.squeeze( azInfDat[dim0].to_dict()[dim1][dim2][dim3][0],2 ) ),
                               kde_bandwidth,n_draws,random_seed+i)[0] for i in range(0,n_chains)]), axis=2) } )
        else:
            print('Re-scaling of "'+dim2+'" not yet supported. '+\
                  'np.shape(azInfDat[dim0].to_dict()[dim1][dim2][dim3])[3] in not 1!')
    """
    from scipy.stats import gaussian_kde
    import xarray as xr
    import numpy as np
    # azInfDat = tests_idata_i arviz.InferenceData
    # dim1 e.g. 'data_vars'
    # dim2 e.g. 'test:intervention'
    n_chains = azInfDat.posterior.dims['chain']
    n_draws = azInfDat.posterior.dims['draw']
    reshape_fun = lambda mat,bw,n,random_seed: gaussian_kde(mat,bw_method=kde_bandwidth).resample(n,random_seed)
    if len(azInfDat[dim0].to_dict()[dim1][dim2]['dims']) == 2:
        return xr.DataArray.from_dict({
            "dims":  azInfDat[dim0].to_dict()[dim1][dim2]['dims'],
            "attrs": {},
            "coords":azInfDat.posterior[dim2].to_dict()['coords'],
            "name":  dim2,
            "data":  np.array([np.apply_along_axis(reshape_fun,0,
                               azInfDat[dim0].to_dict()[dim1][dim2][dim3][0],
                               kde_bandwidth,n_draws,random_seed+i)[0] for i in range(0,n_chains)])} )
    elif len(azInfDat[dim0].to_dict()[dim1][dim2]['dims']) == 3:
        return xr.DataArray.from_dict({
            "dims":  azInfDat[dim0].to_dict()[dim1][dim2]['dims'],
            "attrs": {},
            "coords":azInfDat.posterior[dim2].to_dict()['coords'],
            "name":  dim2,
            "data":  np.array([np.apply_along_axis(reshape_fun,0,
                               np.matrix(azInfDat[dim0].to_dict()[dim1][dim2][dim3][0]),
                               kde_bandwidth,n_draws,random_seed+i)[0] for i in range(0,n_chains)])} )
    else:
        print('Re-scaling of "'+dim2+'" not yet supported.')

def xr_ajdust_dims(azInfDat,dim0='prior',dim1='dims'):
    adj = ['chain','draw']
    upd_dims = azInfDat[dim0].to_dict()[dim1]
    for dim2 in adj:
        upd_dims[dim2] = azInfDat.posterior.to_dict()[dim1][dim2]
    return upd_dims

def xr_ajdust_coords(azInfDat,dim0='prior',dim1='coords'):
    import numpy as np
    adj = ['chain','draw']
    upd_coords = azInfDat[dim0].to_dict()[dim1]
    for c in adj:
        upd_coords[c] = azInfDat.posterior.to_dict()[dim1][c]
    return {k: np.array(upd_coords[k]['data'],dtype=object) for k in upd_coords.keys()}

def xr_adjust_data_vars(azInfDat,dim0='prior',dim1='data_vars',kde_bandwidth = 0.001):
    import numpy as np
    import xarray as xr
    from scipy.stats import gaussian_kde
    d = dict()
    for dim2 in azInfDat[dim0].to_dict()[dim1].keys():
        if len(azInfDat[dim0].to_dict()[dim1][dim2]['dims'])<4:
            d[dim2]=xr.DataArray.from_dict({'data':xr_resample_data(azInfDat,dim1,dim2),
                    'dims':azInfDat.posterior.to_dict()[dim1][dim2]['dims']})
    # add scaled value mean from prior predictive too
    n_chains = azInfDat.posterior.dims['chain']
    n_draws = azInfDat.posterior.dims['draw']
    dim0 = 'prior_predictive'
    dim2 = 'scaled_value'
    dim3 = 'data'
    random_seed = 1234
    reshape_fun = lambda mat,bw,n,random_seed: gaussian_kde(mat,bw_method=kde_bandwidth).resample(n,random_seed)
    d['scaled_value_mean'] = xr.DataArray.from_dict({
            "dims":  azInfDat.posterior.to_dict()[dim1][dim2+'_mean']['dims'],
            "attrs": {},
            "coords":azInfDat.posterior[dim2+'_mean'].to_dict()['coords'],
            "name":  dim2+'_mean',
            "data":  np.array([np.apply_along_axis(reshape_fun,0,
                               np.matrix(azInfDat[dim0].to_dict()[dim1][dim2][dim3][0]),
                               kde_bandwidth,n_draws,random_seed+i)[0] for i in range(0,n_chains)])})
    return d

def az_adjust_prior_to_posterior(azInfDat,dim0='prior'):
    #add_prior_adj2post_to_azInfDat(global_idata_s)
    upd_attrs = azInfDat[dim0].to_dict()['attrs']
    upd_attrs['chains and draws like posterior']='Yes'
    return {
        'coords': xr_ajdust_coords(azInfDat),
        'attrs':upd_attrs,
        'dims':xr_ajdust_dims(azInfDat),
        'data_vars':xr_adjust_data_vars(azInfDat)}

def rem_dims_last(prior_resamp):
    rename_dims_dict = dict()
    update_coords_dict = dict()
    for k in prior_resamp.to_dict()['dims'].keys():
        if k.split('_')[-1]=='0':
            rename_dims_dict[k]='_'.join(k.split('_')[0:-1])
    return rename_dims_dict

def add_prior_adj2post_to_azInfDat(azInfDat):
    import arviz as az
    prior_resamp = az.from_dict(xr_adjust_data_vars(azInfDat)).posterior
    prior_resamp = prior_resamp.rename(rem_dims_last(prior_resamp))
    prior_resamp = prior_resamp.rename({'scaled_value_mean_dim':'scaled_value_dim_0'})
    prior_resamp = prior_resamp.assign_coords(xr_ajdust_coords(azInfDat))
    return azInfDat.add_groups(prior_adj2post = prior_resamp)
    
def prior_prob_BF(idat, key, lvl_h0, lvl_h1, fig, ax, stepsize = 0.001,figsize=(4.5,4.5)):
	"""
	self data selecting version
	"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    p_pris = np.arange(10**(-16),1+stepsize,stepsize)
    vals = []
    for p_pri in p_pris:
        vals.append(az_bf(idat, key=key, lvl_h0=lvl_h0, lvl_h1=lvl_h1, prior_sim=1-p_pri, prior_diff=p_pri).loc['pH1gD',:].to_list()[0])
    bf = az_bf(idat, key=key, lvl_h0=lvl_h0, lvl_h1=lvl_h1, prior_sim=0.5, prior_diff=0.5).loc['bf10',:].to_list()[0]
    p05,p95 = get_prior_probs(vals,p_pris)
    if (p95 > 1) or (p95 < 0):
        p95 = np.nan
    if (p05 > 1) or (p05 < 0):
        p05 = np.nan
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
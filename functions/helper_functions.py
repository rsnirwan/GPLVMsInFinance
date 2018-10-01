import pystan
import pickle
from hashlib import md5
from zipfile import ZipFile
import pandas as pd
import numpy as np
import os
from scipy.optimize import minimize


# numbers in model_dict are based on the function 'kernel_f' in stan_gplvm_covariance.txt
# adjust accordingly
model_dict = {'linear': 0,
               'squared_exp': 1,
               'exp': 2,
               'matern32': 3,
               'matern52': 4,
               'squared_exp_m_linear': 5,
               'exp_m_linear': 6,
               'squared_exp_p_linear': 7,
               'exp_p_linear': 8,
             }


def vb(data_dict, stan_model, init='random', iter=10000, tries=5, num=0):
    """
        vb - variational bayes
        Approximates the posterior with independent gaussians \
        and returns samples from the gaussians. 
    """
    try:
        fit = stan_model.vb(data=data_dict, diagnostic_file='d_{}.csv'.format(num),
                       sample_file='s_{}.csv'.format(num), elbo_samples=100, init=init,
                       iter=iter)
        diagnostic = pd.read_csv('d_{}.csv'.format(num), 
                                 names=['iter', 'time_in_seconds', 'ELBO'], 
                                 comment='#', sep=',')
        sample = pd.read_csv('s_{}.csv'.format(num), comment='#', sep=',')
        #print('vb - ELBO: {}'.format(diagnostic.loc[:,'ELBO'].values[-1]))
        os.remove('d_{}.csv'.format(num))
        os.remove('s_{}.csv'.format(num))
    except pd.errors.ParserError:
        print('pandas ParserError - trying again.')
        diagnostic, sample = vb(data_dict, stan_model, init=init, 
                                iter=iter, tries=tries, num=num)
        
    for _ in range(tries-1):
        diagnostic_, sample_ = vb(data_dict, stan_model, init, 
                                  iter=iter, tries=1, num=num)
        if diagnostic.loc[:,'ELBO'].values[-1] < diagnostic_.loc[:,'ELBO'].values[-1]:
            diagnostic = diagnostic_
            sample = sample_
        del(diagnostic_, sample_)
    return diagnostic, sample


def get_rebalancing_times(start, end, window, delta):
    """
    outputs a list with dates between start and end in a 
    stepsize of delta starting by start+window
    """
    timesteps = map(lambda x: delta*x, range(0,int((end-start)/delta)+1))
    timesteps = [start+window+i for i in timesteps if start+window+i < end-delta]
    return timesteps


def rand_weights(n):
    k = np.random.rand(n)
    return k/sum(k)
    
    

def error_function_minimal_variance(x, *args):  
    """
    minimizes x^T*cov*x - cov is args[0]
    """
    x = np.array(x).reshape(-1,1)
    return x.T.dot( args[0].dot(x) ).squeeze() 



def minimal_variance_portfolio(returns, cov, ret=None):
    ''' 
    input: returns - in format: number stocks x number days
    output: min(w^T*cov*w) 
       - can be extended to min(w^T*cov*w-mu*returns) [change error-function as well]
    '''
    n = len(cov)#len(returns)
    S = cov#np.cov(returns)
    
    def constraint1(x):
        return np.array(x).sum() - 1
    
    constraints = [{'type':'eq', 'fun':constraint1},]
    bounds = [(0,.1) for i in range(n)]
    
    x0 = rand_weights(n)
    outp = minimize(error_function_minimal_variance, x0, args=(S,), method='SLSQP', 
                    constraints=constraints, bounds=bounds)
    portfolio = outp['x'].reshape(-1,1)
    error2 = outp['fun']

    for i in range(10):
        x0 = rand_weights(n)
        outp = minimize(error_function_minimal_variance, x0, args=(S,), method='SLSQP', 
                        constraints=constraints, bounds=bounds, options={'ftol': 1e-9})
        error = outp['fun']
        if error < error2:
            portfolio = outp['x'].reshape(-1,1)
            error2 = error
        #print(np.sqrt(error))
    
    return np.array(portfolio).squeeze()




def get_stock_list_random(N, start, path_zip_file):
    #zf = ZipFile('../snpStockData.zip')
    zf = ZipFile(path_zip_file)
    snp_info = pd.read_csv(zf.open('snpStockData/0snp500info.csv'), 
                           parse_dates=['start at yahoo'])

    #prior 20.. GOOGL is a copy of GOOG, datareader was unable to read BCR 
    #not enough data for CTXS, BHF
    exclude_stocks = ['GOOGL', 'BCR', 'CTXS', 'BHF']

    for stock in exclude_stocks:
        snp_info.drop(index=snp_info.where(snp_info['Ticker symbol'] == stock) \
                      .dropna().index, inplace=True)

    # get ride of stocks if their data is not available in the specified period
    snp_info = snp_info.loc[snp_info.loc[:,'start at yahoo'] < start, :]
    list_available_stocks = snp_info.loc[:, 'Ticker symbol'].values

    # get N random stocks form available stocks
    stock_list = list_available_stocks.copy()
    np.random.shuffle(stock_list)
    return stock_list[:N]



def get_return_price(stock_list, start, end, path_zip_file):
    zf = ZipFile(path_zip_file)
    df_list = []
    for stock in stock_list:
        # read Adjusted Close prices from start to end and append to df_list
        tmp = pd.read_csv(zf.open('snpStockData/{}.csv'.format(stock)), 
                          usecols=['Adj Close', 'Date'],
                          index_col='Date',
                          parse_dates=['Date']).loc[start:end, 'Adj Close']
        df_list.append( tmp )
    
    # concat all dfs in df_list in a new dataframe with date as first axes
    # and stock as second axes  
    # return returns of the prices
    return pd.concat(df_list, axis=1, keys=stock_list).pct_change().iloc[1:]



def get_sector_assignment_dict(path_zip_file, stock_list):
    # return a list of assigned sectors to stocks in stock_list
    zf = ZipFile(path_zip_file)
    snp_info = pd.read_csv(zf.open('snpStockData/0snp500info.csv'), 
                           parse_dates=['start at yahoo'])
    di = dict(snp_info.loc[:, ['Ticker symbol', 'GICS Sector']].values)
    return [di[stock] for stock in stock_list]



# from: http://pystan.readthedocs.io/en/latest/avoiding_recompilation.html
def StanModel_cache(model_code, model_name=None, **kwargs):
    """Use just as you would `stan`"""
    code_hash = md5(model_code.encode('ascii')).hexdigest()
    if model_name is None:
        cache_fn = 'cached-model-{}.pkl'.format(code_hash)
    else:
        cache_fn = 'cached-{}-{}.pkl'.format(model_name, code_hash)
    try:
        sm = pickle.load(open(cache_fn, 'rb'))
    except:
        sm = pystan.StanModel(model_code=model_code)
        with open(cache_fn, 'wb') as f:
            pickle.dump(sm, f)
    else:
        print("Using cached StanModel")
    return sm

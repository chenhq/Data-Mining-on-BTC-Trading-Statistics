import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr


'''这两个暂时不管'''
def BANCHMARKINDEXCLOSE():
    return np.nan

def BANCHMARKINDEXOPEN():
    return np.nan


#RET,DTM和DBM我就直接在ipython notebook中定义过了
'''
def RET_func(CLOSE):
#here I first use forwardfill then backfill to fill the NAN values, 
#because the first digit of this DataFrame is NaN, so bfill method is required here.
    return (CLOSE/CLOSE.shift(1) - 1).fillna(method = 'ffill').fillna(method='bfill')

def DTM(OPEN, HIGH):
    return (OPEN.values <= DELAY(OPEN, 1).values) \
            * MAX(HIGH - OPEN, OPEN - DELAY(OPEN, 1))
    
def DBM(OPEN, LOW):
    return (OPEN >= DELAY(OPEN, 1).values) \
            * MAX(OPEN - LOW, OPEN - DELAY(OPEN, 1))
'''
#我在原文中直接重新写出了TR，HD和LD的表达式
'''
def TR():
    return MAX(MAX(HIGH-LOW,ABS(HIGH-DELAY(CLOSE,1))),ABS(LOW-DELAY(CLOSE,1)) )

def HD():
    return HIGH-DELAY(HIGH,1)

def LD(Data):
    return DELAY(LOW,1)-LOW
'''

'''Fama French 三因子 这里暂时也不做计算'''
def HML():
    return np.nan

def SMB():
    return np.nan

def MKE():
    return np.nan

'''Special Variable，这里暂时也不做计算'''
def SELF():
    return np.nan

'''rank输出的序数然后embed到(0,1]上'''
def RANK(A):
    return A.rank(ascending = True)/len(A)

#def MAX(A, B):
#    return pd.DataFrame({'A':A.values, 'B':B.values}, index = A.index).max(axis=1)

def MAX(A, B):
    return ((A > B) * A + (~(A >= B)) * B)


def MIN(A, B):
    return ((A > B) * B + (~(A >= B)) * A)


def STD(A, n):
    return A.rolling(window = n, min_periods = 1).std().fillna(method = 'bfill')
#   pd.rolling_std(A, window = n, min_periods = 1).fillna(method = 'ffill').fillna(method='bfill')

def CORR(A, B, n):
    cache = A.rolling(window = n, min_periods = 1).corr(other=B).fillna(method = 'ffill').fillna(method='bfill')
    cache = cache.replace(to_replace = [np.inf, -np.inf, np.nan], value = 0)
    return cache

def DELTA(A, n):
    return (A - DELAY(A, delay=n)).fillna(method = 'ffill').fillna(method = 'bfill')

def LOG(A):
    A[A<0.00001] = 0.00001
    return np.log(A)

def SUM(A, n):
    return A.rolling(window = n, min_periods = 1, center = False).sum().fillna(method = 'ffill').fillna(method='bfill')

def ABS(A):
    return np.abs(A)

def MEAN(A, n):
	return A.rolling(window = n, min_periods = 1, center = False).mean().fillna(method = 'ffill').fillna(method='bfill')
    

def SKEW(A, n):
    return A.rolling(window = n, min_periods = 1, center = False).skew().fillna(method = 'ffill').fillna(method='bfill')

def KURT(A, n):
    return A.rolling(window = n, min_periods = 1, center = False).kurt().fillna(method = 'ffill').fillna(method='bfill')

def QUANTILE(A, n, d):
    if d > 100:
        d = 100
    elif d < 0:
        d = 0
    return A.rolling(window = n, min_periods = 1, center = False).apply(lambda x:np.percentile(x,d)).fillna(method = 'ffill').fillna(method='bfill')


'''
TSRANK(A, n)的功能是求得序列A的每个当前值在过去n天的顺序排位，
因此这里不能设置 min_periods = 1

归一化后的也可以设置
'''
def TSRANK(A, n):
    return A.rolling(window=n, center = False).apply(lambda x: np.argsort(x).tolist().index(n-1)/n).fillna(method = 'ffill').fillna(method='bfill')

def SIGN(A):
    return np.sign(A)

def SIGN_POWER(A, d):
    return (np.sign(A) * np.abs(A)**d).fillna(method = 'ffill').fillna(method = 'bfill')

def COVIANCE(A, B, n):
    cache =  A.rolling(window=n, min_periods=1).cov(other= B).fillna(method='ffill')
    cache = cache.replace([np.inf, -np.inf, np.nan], 0)
    return cache

    
def DELAY(A, delay=1):
    if delay < 0:
        delay = 0
    return A.shift(delay).fillna(method = 'pad').fillna(method = 'bfill')

def TSMIN(A, n):
    return A.rolling(window = n, min_periods = 1, center = False).min()

def TSMAX(A, n):
    return A.rolling(window = n, min_periods = 1, center=False).max()

def PROD(A, n):
    return A.rolling(window=n, center=False, min_periods = 1).apply(lambda x: np.prod(x))

def COUNT(condition, n):
    cache = (condition).fillna(0)#now condition is the boolean DataFrame/Series/Array
    return cache.rolling(window = n, center = False, min_periods = 1).sum()


'''终于实现了 哈哈哈.........
from sklearn import linear_model

def REGBETA(A, B, n):
    out_dates = []
    out_beta = []

    model_ols = linear_model.LinearRegression()

    for _start in range(0, len(A.values)-n):        
        _end = _start + n
        model_ols.fit(A.values[_start:_end].reshape(len(A.values[_start:_end]),-1),\
                      B.values[_start:_end].reshape(len(B.values[_start:_end]),-1))

        #store output
        out_dates.append(A.index[_end])
        out_beta.append(model_ols.coef_[0][0])

    return pd.DataFrame({'beta':out_beta}, index=out_dates)



def REGRESI(A, B, n):
    out_dates = []
    out_beta = []

    model_ols = linear_model.LinearRegression()

    for _start in range(0, len(A.values)-n):        
        _end = _start + n
        model_ols.fit(A.values[_start:_end].reshape(len(A.values[_start:_end]),-1),\
                      B.values[_start:_end].reshape(len(B.values[_start:_end]),-1))

        #store output
        out_dates.append(A.index[_end])
        out_beta.append(model_ols.coef_[0][0])

    return pd.DataFrame({'residual':out_beta}, index=out_dates)
'''



def ZSCORE(A, n):
    zsocre = A.rolling(window = n, center = False).apply(lambda x: (x[-1] - np.mean(x)) / np.std(x))
    if n <= 1:
        return zsocre.fillna(1)
    else:
        return zsocre.fillna(method = 'ffill').fillna(method = 'bfill')


#regbeta 这里不能使用min_periods，因为B是fixed window length
def REGBETA(A, B, n):
    assert (n == len(B))
    regbeta = A.rolling(window = n, center = False).apply(lambda x: np.cov(x, B)[0][1]/np.var(B))
    return regbeta
#np.cov(x, B)[0][1] == np.cov(x, B)[1][0]，这是一个对称阵


def REGRESI(A, B, n):
    return B - A * REGBETA(A, B, n)

#这里我加了一个assert，从直观意义出发
def SMA(A, n, m):
    assert n >= m #当n=m时SMA返回结果与原序列一致
    cache = A.values
    for i in range(1,len(A)):
        A.values[i] = (A.values[i-1]*(n-m) + cache[i-1])/n
    return A.fillna(method = 'ffill').fillna(method = 'bfill')


def SUMIF(A, n, condition):
    return (A*condition).rolling(window = n, center = False, min_periods = 1).sum().fillna(method = 'ffill').fillna(method='bfill')


'''
报告中描述WMA()功能的原话是：
计算A前n期样本加权平均值，权重为0.9i，(i表示样本距离当前时点的间隔)
这个权值好像没有归一化，以下是按照归一化的结果写的


更新后变为coef/100，而不是简单的0.9
'''

def WMA(A, n):
    w = np.arange(1, n+1) * 0.9
    w = w/w.sum()
    return A.rolling(n).apply(lambda x: (x * w).sum()).fillna(method = 'ffill').fillna(method='bfill')
    
def DECAYLINEAR(A, n):
    w = np.arange(n,0,-1) 
    w = w/w.sum()
    return A.rolling(n).apply(lambda x: (x * w).sum()).fillna(method = 'ffill').fillna(method='bfill')

def DECAYEXP(A, n, coef):
    w = np.arange(1, n+1) * (coef / 100.)
    w = w/w.sum()
    return A.rolling(n).apply(lambda x: (x * w).sum()).fillna(method = 'ffill').fillna(method='bfill')    

def FILTER(A, condition):
    return A[condition]


def HIGHDAY(A, n):
    return (n-1) - A.rolling(window=n).apply(lambda x: np.argsort(x)[n-1]).fillna(method = 'ffill').fillna(method='bfill')


#报告中写错了，这里应该是计算最小值到当前窗口的距离，因此作出调整   
def LOWDAY(A, n):
    return (n-1) - A.rolling(window=n).apply(lambda x: np.argsort(x)[0]).fillna(method = 'ffill').fillna(method='bfill')

def SEQUENCE(m):
    return np.arange(1,m+1)
    
def SUMAC(A, n):
    return A.rolling(min_periods=1, window=n).sum().fillna(method = 'ffill').fillna(method='bfill')


#---后面增加的是一些我们WorldQuant以及JoinQuant101会用到的函数---


def TS_ARGMIN(A, n):
    return A.rolling(window=n).apply(lambda x: np.argsort(x)[0]).fillna(method = 'ffill').fillna(method='bfill')

def TS_ARGMAX(A, n):
    return A.rolling(window=n).apply(lambda x: np.argsort(x)[n-1]).fillna(method = 'ffill').fillna(method='bfill')


#这里的ADV返回的是这一个期货在最近n天交易的平均价格
def ADV(n):
    return np.nan
    
# here is the function designed for dealing with np.inf
def process_inf(alpha):
    if ((alpha == np.inf) | (alpha == -np.inf)).any():
        print ('DataFrame has infinity values')
    upper_bar = alpha.replace([np.inf, -np.inf], np.nan).dropna(how='all').max()
    lower_bar = alpha.replace([np.inf, -np.inf], np.nan).dropna(how='all').min()
    
    alpha = alpha.replace(to_replace = np.inf, value = upper_bar)
    alpha = alpha.replace(to_replace = -np.inf, value = lower_bar)
    return alpha


def perasonr_hand(A, B):
    A_new = process_inf(A)
    B_new = process_inf(B)
    return pearsonr(A_new, B_new)

def spearmanr_hand(A, B):
    A_new = process_inf(A)
    B_new = process_inf(B)
    return spearmanr(A_new, B_new)




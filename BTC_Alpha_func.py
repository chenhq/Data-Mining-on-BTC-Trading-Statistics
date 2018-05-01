import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

# volume weighted average price
#def VWAP():
#    return amount_/volume_

'''交易量乘上价格，乘以close'''
def AMOUNT():
    return VOLUME_Traded * CLOSE


'''这两个暂时不管'''
def BANCHMARKINDEXCLOSE():
    return np.nan

def BANCHMARKINDEXOPEN():
    return np.nan


def RET(CLOSE):
#here I first use forwardfill then backfill to fill the NAN values, 
#because the first digit of this DataFrame is NaN, so bfill method is required here.
    return (CLOSE/CLOSE.shift(1) - 1).fillna(method = 'ffill').fillna(method='bfill')


#DTM和DBM我就直接在ipython notebook中定义过了
'''
def DTM(OPEN, HIGH):
    return (OPEN.values <= DELAY(OPEN, 1).values) \
            * MAX(HIGH - OPEN, OPEN - DELAY(OPEN, 1))
    
def DBM(OPEN, LOW):
    return (OPEN >= DELAY(OPEN, 1).values) \
            * MAX(OPEN - LOW, OPEN - DELAY(OPEN, 1))
'''
#我在原文中直接重新写出了TR，HD和LD的表达式
def TR():
    return MAX(MAX(HIGH-LOW,ABS(HIGH-DELAY(CLOSE,1))),ABS(LOW-DELAY(CLOSE,1)) )

def HD():
    return HIGH-DELAY(HIGH,1)

def LD(Data):
    return DELAY(LOW,1)-LOW


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

'''?，rank是输出序数、还是直接排序，这里存疑'''
def RANK(Data):
    return Data.rank(ascending = True)

#def MAX(A, B):
#    return pd.DataFrame({'A':A.values, 'B':B.values}, index = A.index).max(axis=1)

def MAX(A, B):
    return ((A > B) * A + ~(A > B) * B)



def MIN(A, B):
#    return pd.DataFrame({'A':A.values, 'B':B.values}, index = A.index).min(axis=1)
    return ((A > B) * B + ~(A > B) * A)



def STD(A, n):
    return A.rolling(window = n, min_periods = 1).std().fillna(method = 'bfill')
#   pd.rolling_std(A, window = n, min_periods = 1).fillna(method = 'ffill').fillna(method='bfill')

def CORR(A, B, n):
    return A.rolling(window = n, min_periods = 1).corr(other=B).fillna(method = 'ffill').fillna(method='bfill')

def DELTA(A, n):
    return A - DELAY(A, delay=n)

def LOG(A):
    return np.log(A)

def SUM(A, n):
    return A.rolling(window = n, min_periods = 1, center = False).sum().fillna(method = 'ffill').fillna(method='bfill')

def ABS(A):
    return np.abs(A)

def MEAN(A, n):
	return A.rolling(window = n, min_periods = 1, center = False).mean().fillna(method = 'ffill').fillna(method='bfill')
    
'''
TSRANK(A, n): 序列 A 的末位值在过去 n 天的顺序排位，这里也不能设置 min_periods = 1
我觉得我这样写跟他的要求一致。。
'''
def TSRANK(A, n):
    return A.rolling(window=n, center = False).apply(lambda x: np.argsort(x).tolist().index(n-1)).fillna(method = 'ffill').fillna(method='bfill')


def SIGN(A):
    return np.sign(A)

def COVIANCE(A, B, n):
    return A.rolling(window=n, min_periods=1).cov(other= B).fillna(method='ffill')
    #return pd.rolling_cov(A, B, window = n, min_periods = 1)

'''为啥还要用.fillna(method = 'pad')???'''
def DELAY(A, delay=1):
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


#typically B is a fixed series
#regbeta 这里不能使用min_periods，因为B是fixed window length
def REGBETA(A, B, n):
    assert (n == len(B))
    regbeta = A.rolling(window = n, center = False).apply(lambda x: np.cov(x, B)[0][1]/np.var(B))
    return regbeta

def REGRESI(A, B, n):
    return B - A * REGBETA(A, B, n)


'''
不清楚Y_i的含义，
定义：SMA(A, n, m) := Y_{i+1} = (A_i * m + Y_i * ( n - m )) / n
示例：alpha9 = SMA(((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))/2)*(HIGH-LOW)/VOLUME,7,2)  
'''

#SMA
def SMA(A, n, m):
    cache = A.values
    for i in range(1,len(A)):
        A.values[i] = (A.values[i-1]*(n-m) + cache[i-1])/n
    return A


def SUMIF(A, n, condition):
    return (A*condition).rolling(window = n, center = False, min_periods = 1).sum().fillna(method = 'ffill').fillna(method='bfill')



'''
关于WMA的原话是：
计算 A 前 n 期样本加权平均值权重为 0.9i，(i 表示样本距离当前时点的间隔)

这个权值好像没有归一化，以下是按照归一化的结果写的
'''
def WMA(A, n):
    w = np.arange(1, n+1) * 0.9
    w = w/w.sum()
    return A.rolling(n).apply(lambda x: (x * w).sum()).fillna(method = 'ffill').fillna(method='bfill')
    
def DECAYLINEAR(A, n):
    w = np.arange(n,0,-1) 
    w = w/w.sum()
    return A.rolling(n).apply(lambda x: (x * w).sum()).fillna(method = 'ffill').fillna(method='bfill')
    

def FILTER(A, condition):
    return A[condition]


def HIGHDAY(A, n):
    return (n-1) - A.rolling(window=n).apply(lambda x: np.argsort(x)[n-1]).fillna(method = 'ffill').fillna(method='bfill')

    
def LOWDAY(A, n):
    return (n-1) - A.rolling(window=n).apply(lambda x: np.argsort(x)[n-1]).fillna(method = 'ffill').fillna(method='bfill')


def SEQUENCE(m):
    return np.array((np.linspace(1,m,m)))
    
def SUMAC(A, n):
    return A.rolling(min_periods=1, window=n).sum().fillna(method = 'ffill').fillna(method='bfill')


# here is the function designed for dealing with np.inf
def process_inf(alpha):
    if ((alpha == np.inf) | (alpha == -np.inf)).any():
        print ('DataFrame has infinity values')
    upper_bar = alpha.replace([np.inf, -np.inf], np.nan).dropna(how='all').max()
    lower_bar = alpha.replace([np.inf, -np.inf], np.nan).dropna(how='all').min()
    
    alpha = alpha.replace(to_replace = np.inf, value = upper_bar)
    alpha = alpha.replace(to_replace = -np.inf, value = lower_bar)
    return alpha


def perasonr(A, B, n):
    A_new = process_inf(A)
    B_new = process_inf(B)

    return A_new.rolling(window = n, min_periods = 1).corr(other=B_new, method='pearson').fillna(method = 'ffill')

def perasonr(A, B, n):
    A_new = process_inf(A)
    B_new = process_inf(B)

    return A_new.rolling(window = n, min_periods = 1).corr(other=B_new, method='spearman').fillna(method = 'ffill')















# coding: utf-8

# 我觉得变量太多的alpha factor我暂时搁置，以及我觉得rank函数比较奇怪，因为range太大了，是否需要设置一个窗口呢？
# 
# 
# ## Dropped Index: 
# - Alpha30(要用到fama三因子)
# - Alpha75(要用到BENCHMARKINDEX)
# - Alpha143(要用到SELF函数)
# - Alpha149(要用到BENCHMARKINDEX)
# - Alpha181(要用到BENCHMARKINDEX)
# - Alpha182(要用到BENCHMARKINDEX)

### 对于：？较为复杂的表达式，我都先用一些中间变量存储中间运算的结果，以下均采用这一做法，不赘述


import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
from scipy.stats.stats import spearmanr
from BTC_Alpha_func import *


def Alpha1(para_list):
    return -1 * CORR(RANK(DELTA(LOG(VOLUME),para_list[0])), RANK((CLOSE-OPEN)/OPEN), para_list[1])

def Alpha2(para_list):
    return (-1 * DELTA((((CLOSE - LOW) - (HIGH - CLOSE)) / (HIGH - LOW)), para_list[0])).fillna(0)


def Alpha3(para_list):
    cache = CLOSE - ((~(CLOSE>DELAY(CLOSE,para_list[0])))*MIN(LOW,DELAY(CLOSE,para_list[0]))\
                   + (~(CLOSE>DELAY(CLOSE,para_list[0])))*MAX(HIGH,DELAY(CLOSE,para_list[0])))
    return SUM((~(CLOSE==DELAY(CLOSE,1)) * cache), para_list[1])


#这里保留1,是因为我觉得Volume/mean(volume,window_size)还是有明确的概念的
def Alpha4(para_list):
    #tail计算的是倒数第二个冒号后面的结果
    tail = (((VOLUME / MEAN(VOLUME,para_list[0])) <= 1) * 1\
         - ~((VOLUME / MEAN(VOLUME,para_list[0])) <= 1) * (-1))
    #med计算的是中间的一个判断句（第1个冒号之后）的结果
    med = ((SUM(CLOSE, para_list[1]) / para_list[1]) < ((SUM(CLOSE, para_list[2]) / para_list[2]) - STD(CLOSE, para_list[2]))) * 1\
     +  ~(((SUM(CLOSE, para_list[1]) / para_list[1]) < ((SUM(CLOSE, para_list[2]) / para_list[2]) - STD(CLOSE, para_list[2])))) * tail
    
    return (((SUM(CLOSE, para_list[2]) / para_list[2]) + STD(CLOSE, para_list[2])) < (SUM(CLOSE, para_list[1]) / para_list[1])) * (-1)\
     +    ~(((SUM(CLOSE, para_list[2]) / para_list[2]) + STD(CLOSE, para_list[2])) < (SUM(CLOSE, para_list[1]) / para_list[1])) * med


def Alpha5(para_list):
    return (-1 * TSMAX(CORR(TSRANK(VOLUME, para_list[0]), TSRANK(HIGH, para_list[0]), para_list[0]), para_list[1]))


#here para_list[0] is a float between(0,1)
def Alpha6(para_list):
    return (RANK(SIGN(DELTA((((OPEN * para_list[0]) + (HIGH * (1.0-para_list[0])))), para_list[1])))* (-1))


def Alpha7(para_list):
    return ((RANK(MAX((VWAP - CLOSE), para_list[0])) + RANK(MIN((VWAP - CLOSE), para_list[0]))) * RANK(DELTA(VOLUME, para_list[0])))


#here para_list[0] is a float between(0,1)
def Alpha8(para_list):
    return RANK(DELTA(((((HIGH + LOW) / 2) * para_list[0]) + (VWAP * (1.0-para_list[0]))), para_list[1]) * -1)

#所有的SMA我都加上了assert，我其实在函数里也已经加上了assert，以下不赘述
def Alpha9(para_list):
    assert para_list[2] <= para_list[1]
    return SMA(((HIGH+LOW)/2-(DELAY(HIGH,para_list[0])+DELAY(LOW,para_list[0]))/2)*(HIGH-LOW)/VOLUME,para_list[1],para_list[2])


#para_list[2] 原来就是平方的，这里先改成了para_list[2]
def Alpha10(para_list):
    return RANK(MAX((STD(RET, para_list[0]) * (RET < 0) + (CLOSE * (~(RET < 0)))**para_list[2], para_list[1])))

def Alpha11(para_list):
    return SUM(((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW)*VOLUME, para_list[0])

def Alpha12(para_list):
    return (RANK((OPEN - (SUM(VWAP, para_list[0]) / para_list[0])))) * (-1 * (RANK(ABS((CLOSE - VWAP)))))

#para_list[0]原来就是开方的，这里也先改了
def Alpha13(para_list):
    return (((HIGH * LOW)**para_list[0]) - VWAP) #这个是取调和平均的 我们就算他不用优化把= =


def Alpha14(para_list):
    return CLOSE-DELAY(CLOSE, para_list[0])

#这里的1.0保留
def Alpha15(para_list):
    return OPEN/DELAY(CLOSE,para_list[0])-1.0

def Alpha16(para_list):
    return (-1 * TSMAX(RANK(CORR(RANK(VOLUME), RANK(VWAP), para_list[0])), para_list[0]))

def Alpha17(para_list):
    return RANK((VWAP - MAX(VWAP, para_list[0])))**(DELTA(CLOSE, para_list[1]))

def Alpha18(para_list):
    return CLOSE/DELAY(CLOSE,para_list[0])

def Alpha19(para_list):
    return (CLOSE <= DELAY(CLOSE,para_list[0])) * (CLOSE - DELAY(CLOSE,para_list[0]))/DELAY(CLOSE,para_list[0])\
      +    (CLOSE >  DELAY(CLOSE,para_list[0])) * (CLOSE - DELAY(CLOSE,para_list[0])/CLOSE)

#100.0保留，表示百分数，以下同
def Alpha20(para_list):
    return (CLOSE-DELAY(CLOSE,para_list[0]))/DELAY(CLOSE,para_list[0])*100.0

def Alpha21(para_list):
    return REGBETA(MEAN(CLOSE,para_list[0]),SEQUENCE(para_list[0]),para_list[0]) 

def Alpha22(para_list):
    return MEAN((CLOSE-MEAN(CLOSE,para_list[0]))/MEAN(CLOSE,para_list[0])\
         -DELAY((CLOSE-MEAN(CLOSE,para_list[0]))/MEAN(CLOSE,para_list[0]),para_list[1]),para_list[2])

def Alpha23(para_list):
    return SMA((CLOSE> DELAY(CLOSE,para_list[0]))*STD(CLOSE,para_list[1]),para_list[1],para_list[2])\
         /(SMA((CLOSE> DELAY(CLOSE,para_list[0]))*STD(CLOSE,para_list[1]),para_list[1],para_list[2])\
          +SMA((CLOSE<=DELAY(CLOSE,para_list[0]))*STD(CLOSE,para_list[1]),para_list[1],para_list[2]))*100.0

def Alpha24(para_list):
    return SMA(CLOSE-DELAY(CLOSE,para_list[0]),para_list[0],para_list[1])

def Alpha25(para_list):
    return ((-1 * RANK((DELTA(CLOSE,para_list[0]) * (1 - RANK(DECAYLINEAR((VOLUME / MEAN(VOLUME,para_list[1])), para_list[2]))))))  * (1.0 + RANK(SUM(RET, para_list[3]))))

def Alpha26(para_list):
    return (((SUM(CLOSE, para_list[0]) / para_list[0]) - CLOSE)) + ((CORR(VWAP, DELAY(CLOSE, para_list[1]), para_list[2])))

def Alpha27(para_list):
    return WMA((CLOSE-DELAY(CLOSE,para_list[0]))/DELAY(CLOSE,para_list[0])*100.0\
              +(CLOSE-DELAY(CLOSE,para_list[1]))/DELAY(CLOSE,para_list[1])*100.0,para_list[2])

#这里的para_list[3]原先设置为9，para_list[4],para_list[5]分别的设置为3和2
def Alpha28(para_list):
    return para_list[4]*SMA((CLOSE-TSMIN(LOW,para_list[0]))/(TSMAX(HIGH,para_list[0])-TSMIN(LOW,para_list[0]))*100,para_list[1],para_list[2])\
      -para_list[5]*SMA(SMA((CLOSE-TSMIN(LOW,para_list[0]))/(MAX(  HIGH,para_list[3])-TSMAX(LOW,para_list[0]))*100,para_list[1],para_list[2]),para_list[1],para_list[2])


def Alpha29(para_list):
    return (CLOSE-DELAY(CLOSE,para_list[0]))/DELAY(CLOSE,para_list[0])*VOLUME

def Alpha30(para_list):
    return CLOSE - CLOSE

def Alpha31(para_list):
    return (CLOSE-MEAN(CLOSE,para_list[0]))/MEAN(CLOSE,para_list[0])*100.0

def Alpha32(para_list):
    return (-1 * SUM(RANK(CORR(RANK(HIGH), RANK(VOLUME), para_list[0])), para_list[0]))


def Alpha33(para_list):
    return ((((-1 * TSMIN(LOW, para_list[0])) + DELAY(TSMIN(LOW, para_list[0]), para_list[0])) * RANK(((SUM(RET, para_list[1]) - SUM(RET, para_list[2])) / (para_list[3]))))* TSRANK(VOLUME, para_list[0]))


def Alpha34(para_list):
    return MEAN(CLOSE,para_list[0])/CLOSE

#here para_list[2] is a float between(0,1)
def Alpha35(para_list):
    return (-MIN(RANK(DECAYLINEAR(DELTA(OPEN, para_list[0]), para_list[1])),\
                 RANK(DECAYLINEAR(CORR((VOLUME), ((OPEN * para_list[2]) + (OPEN *(1-para_list[2]))), para_list[3]),para_list[4]))))

def Alpha36(para_list):
    return RANK(SUM(CORR(RANK(VOLUME), RANK(VWAP), para_list[0]), para_list[1]))

def Alpha37(para_list):
    return (- RANK(((SUM(OPEN, para_list[0]) * SUM(RET, para_list[0]))\
            - DELAY((SUM(OPEN, para_list[0]) * SUM(RET, para_list[0])), para_list[1]))))

def Alpha38(para_list):
    return ((SUM(HIGH, para_list[0])/para_list[0]) < HIGH) * (-1.0 * DELTA(HIGH, para_list[1]))

def Alpha39(para_list):
    return (-(RANK(DECAYLINEAR(DELTA((CLOSE), para_list[0]),para_list[1]))\
             -RANK(DECAYLINEAR(CORR(((VWAP * para_list[2]) + (OPEN * (1-para_list[2]))), SUM(MEAN(VOLUME,para_list[3]), para_list[4]), para_list[5]), para_list[6]))))

def Alpha40(para_list):
    return SUM((CLOSE > DELAY(CLOSE,para_list[0]))*VOLUME, para_list[1])\
          /SUM((CLOSE<= DELAY(CLOSE,para_list[0]))*VOLUME, para_list[1])*100.0

def Alpha41(para_list):
    return (RANK(-MAX(DELTA((VWAP), para_list[0]), para_list[1])))

def Alpha42(para_list):
    return ((-RANK(STD(HIGH, para_list[0]))) * CORR(HIGH, VOLUME, para_list[0]))

def Alpha43(para_list):
    return SUM(VOLUME * (CLOSE>DELAY(CLOSE,para_list[0]))\
              -VOLUME *(~(CLOSE>DELAY(CLOSE,para_list[0]))) * (CLOSE<DELAY(CLOSE,para_list[0])), para_list[1])

def Alpha44(para_list):
    return  TSRANK(DECAYLINEAR(CORR(LOW, MEAN(VOLUME,para_list[0]), para_list[1]), para_list[2]), para_list[3])\
          + TSRANK(DECAYLINEAR(DELTA(VWAP, para_list[4]), para_list[5]), para_list[6])

def Alpha45(para_list):
    return RANK(DELTA(CLOSE * para_list[0] + OPEN * (1-para_list[0]), para_list[1]))\
         * RANK(CORR(VWAP, MEAN(VOLUME, para_list[2]), para_list[3]))

#这里4.0也有很明确的概念，就是表示4个window的平均值
def Alpha46(para_list):
    return (MEAN(CLOSE,para_list[0])\
          + MEAN(CLOSE,para_list[1])\
          + MEAN(CLOSE,para_list[2])\
          + MEAN(CLOSE,para_list[3]))/(4.0*CLOSE)

def Alpha47(para_list):
    return SMA((TSMAX(HIGH,para_list[0])-CLOSE)/(TSMAX(HIGH,para_list[0]) - TSMIN(LOW,para_list[0]))*100.0, para_list[1], para_list[2])

def Alpha48(para_list):
    return  (-(RANK(SIGN(CLOSE - DELAY(CLOSE, para_list[0]))\
                  + SIGN(DELAY(CLOSE, para_list[0]) - DELAY(CLOSE, para_list[1]))\
                  + SIGN(DELAY(CLOSE, para_list[1]) - DELAY(CLOSE, para_list[2])))\
         * SUM(VOLUME, para_list[1] + para_list[2])) / SUM(VOLUME, para_list[3]))


def Alpha49(para_list):
    dividend = SUM(MAX(ABS(HIGH-DELAY(HIGH,para_list[0])),ABS(LOW-DELAY(LOW,para_list[0]))),para_list[1])
    divisor =  SUM(~((HIGH+LOW) >= (DELAY(HIGH,para_list[0]) + DELAY(LOW,para_list[0])))\
                  *MAX(ABS(HIGH-DELAY(HIGH,para_list[0])),ABS(LOW-DELAY(LOW,para_list[0]))),para_list[1])

    return divisor/dividend


def Alpha50(para_list):
    subtend = SUM(~((HIGH+LOW) <= (DELAY(HIGH,para_list[0]) + DELAY(LOW,para_list[0])))\
                 *MAX(ABS(HIGH-DELAY(HIGH,para_list[0])),ABS(LOW-DELAY(LOW,para_list[0]))),para_list[1])\
            /(SUM(MAX(ABS(HIGH-DELAY(HIGH,para_list[0])),ABS(LOW-DELAY(LOW,para_list[0]))),para_list[1]))
    
    minuend = SUM(~((HIGH+LOW) >= (DELAY(HIGH,para_list[0]) + DELAY(LOW,para_list[0])))\
                 *MAX(ABS(HIGH-DELAY(HIGH,para_list[0])),ABS(LOW-DELAY(LOW,para_list[0]))),para_list[1])\
            /(SUM(MAX(ABS(HIGH-DELAY(HIGH,para_list[0])),ABS(LOW-DELAY(LOW,para_list[0]))),para_list[1]))

    return subtend - minuend   


def Alpha51(para_list):
    return SUM(~((HIGH+LOW) <= (DELAY(HIGH,para_list[0]) + DELAY(LOW,para_list[0])))\
              *MAX(ABS(HIGH-DELAY(HIGH,para_list[0])),ABS(LOW-DELAY(LOW,para_list[0]))),para_list[1])\
         /(SUM(MAX(ABS(HIGH-DELAY(HIGH,para_list[0])),ABS(LOW-DELAY(LOW,para_list[0]))),para_list[1]))

def Alpha52(para_list):
    return SUM(MAX(0, HIGH-DELAY((HIGH+LOW+CLOSE)/3,para_list[0])), para_list[1])\
          /SUM(MAX(0, DELAY((HIGH+LOW+CLOSE)/3,para_list[0]) - LOW),para_list[1])* 100.0


def Alpha53(para_list):
    return COUNT(CLOSE>DELAY(CLOSE,para_list[0]),para_list[1])/para_list[1]*100.0

def Alpha54(para_list):
    return (-RANK((STD(ABS(CLOSE - OPEN), para_list[0]) + (CLOSE - OPEN)) + CORR(CLOSE, OPEN, para_list[0])))


#part_B1_value中有/2，/4算是decay sum吧。。，我也替换成了两个参数
def Alpha55(para_list):
    part_C_value = MAX(ABS(HIGH-DELAY(CLOSE,para_list[0])),\
                       ABS(LOW- DELAY(CLOSE,para_list[0])))

    part_A_value = (CLOSE+(CLOSE-OPEN)/2.0-DELAY(OPEN,para_list[0]))

    part_B1_cond = (ABS(HIGH-DELAY(CLOSE,para_list[0])) > ABS(LOW -DELAY(CLOSE,para_list[0])))\
                  &(ABS(HIGH-DELAY(CLOSE,para_list[0])) > ABS(HIGH-DELAY(LOW,  para_list[0])))

    part_B2_cond = (ABS(LOW- DELAY(CLOSE,para_list[0])) > ABS(HIGH-DELAY(LOW,  para_list[0])))\
                  &(ABS(LOW- DELAY(CLOSE,para_list[0])) > ABS(HIGH-DELAY(CLOSE,para_list[0])))
    
    part_B1_value=  ABS(HIGH-DELAY(CLOSE,para_list[0])) + ABS(LOW -DELAY(CLOSE,para_list[0]))/para_list[1]\
                  + ABS(DELAY(CLOSE,para_list[0])-DELAY(OPEN, para_list[0]))/para_list[2]
    
    part_B2nvalue= (ABS(HIGH-DELAY(LOW ,para_list[0]))  + ABS(DELAY(CLOSE,para_list[0])-DELAY(OPEN,para_list[0]))/para_list[2])
    
    part_B_value = (part_B1_cond | (~part_B1_cond) & part_B2_cond) * part_B1_value\
               + ((~part_B1_cond) & (~part_B2_cond)) * part_B2nvalue
    
    return SUM(part_A_value/part_B_value*part_C_value, para_list[1])
   

#这个signal是返回一个bool list，与原文对照过了，表达式一致，很迷
def Alpha56(paralist):
    return RANK((OPEN - TSMIN(OPEN, para_list[0]))) < RANK((RANK(CORR(SUM(((HIGH + LOW)/2.0), para_list[1]), SUM(MEAN(VOLUME,para_list[2]), para_list[3]), para_list[4]))**para_list[5]))

def Alpha57(para_list):
    return SMA((CLOSE-TSMIN(LOW,para_list[0]))/(TSMAX(HIGH,para_list[0])-TSMIN(LOW,para_list[0])),para_list[1],para_list[2])

def Alpha58(para_list):
    return COUNT(CLOSE>DELAY(CLOSE,para_list[0]),para_list[1])/para_list[1]

def Alpha59(para_list):
    return SUM((CLOSE!=DELAY(CLOSE,para_list[0]))*CLOSE\
            - ((CLOSE>DELAY(CLOSE,para_list[0]))* MIN(LOW, DELAY(CLOSE,para_list[0]))\
            + ~(CLOSE>DELAY(CLOSE,para_list[0]) * MAX(HIGH,DELAY(CLOSE,para_list[0])))), para_list[1])

def Alpha60(para_list):
    return SUM(((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW)*VOLUME,para_list[0])

def Alpha61(para_list):
    return (-MAX(RANK(DECAYLINEAR(DELTA(VWAP,para_list[0]),para_list[1])),\
                 RANK(DECAYLINEAR(RANK(CORR(LOW,MEAN(VOLUME,para_list[2]), para_list[3])),para_list[4]))))

def Alpha62(para_list):
    return (-CORR(HIGH, RANK(VOLUME), para_list[0]))


def Alpha63(para_list):
    return (SMA(MAX(CLOSE-DELAY(CLOSE,para_list[0]),0),para_list[1],para_list[2])\
           /SMA(ABS(CLOSE-DELAY(CLOSE,para_list[0]))  ,para_list[1],para_list[2]))


def Alpha64(para_list):
    return -MAX(RANK(DECAYLINEAR(CORR(RANK(VWAP), RANK(VOLUME), para_list[0]),para_list[0])),\
                RANK(DECAYLINEAR(MAX(CORR(RANK(CLOSE), RANK(MEAN(VOLUME,para_list[1])), para_list[0]), para_list[2]), para_list[3])))

def Alpha65(para_list):
    return MEAN(CLOSE,para_list[0])/CLOSE

def Alpha66(para_list):
    return (CLOSE-MEAN(CLOSE,para_list[0]))/MEAN(CLOSE,para_list[0])

def Alpha67(para_list):
    return SMA(MAX(CLOSE-DELAY(CLOSE,),0),para_list[1],para_list[2])\
          /SMA(ABS(CLOSE-DELAY(CLOSE,para_list[0])),para_list[1],para_list[2])

def Alpha68(para_list):
    return SMA(((HIGH+LOW)/2-(DELAY(HIGH,para_list[0])+DELAY(LOW,para_list[0]))/para_list[0])*(HIGH-LOW)/VOLUME,para_list[1],para_list[2])

def Alpha69(para_list):
    cache= (SUM(DTM,para_list[0])>SUM(DBM,para_list[0])) * (SUM(DTM,para_list[0])- SUM(DBM,para_list[0]))/SUM(DTM,para_list[0])         +(~(SUM(DTM,para_list[0])>SUM(DBM,para_list[0])) & (SUM(DTM,para_list[0])!=SUM(DBM,para_list[0]))                                                         * (SUM(DTM,para_list[0])- SUM(DBM,para_list[0]))/SUM(DBM,para_list[0]))
    return cache.fillna(method='ffill').fillna(method='bfill')

def Alpha70(para_list):
    return STD(AMOUNT,para_list[0])

def Alpha71(para_list):
    return (CLOSE-MEAN(CLOSE,para_list[0]))/MEAN(CLOSE,para_list[0])

def Alpha72(para_list):
    return SMA((TSMAX(HIGH,para_list[0])-CLOSE)/(TSMAX(HIGH,para_list[0])-TSMIN(LOW,para_list[0])),para_list[1],para_list[2])

def Alpha73(para_list):
    return (TSRANK(DECAYLINEAR(DECAYLINEAR(CORR(CLOSE, VOLUME,para_list[0]),para_list[1]),para_list[2]),para_list[3])-RANK(DECAYLINEAR(CORR(VWAP, MEAN(VOLUME,30),4),3))) * -1

#para_list[0] is a float between (0,1)
def Alpha74(para_list):
    return RANK(CORR(SUM(((LOW * para_list[0]) + VWAP*(1-para_list[0])), para_list[1]), SUM(MEAN(VOLUME,para_list[2]),para_list[1]), para_list[3])) + RANK(CORR(RANK(VWAP), RANK(VOLUME), para_list[4]))

def Alpha75(para_list):
    return CLOSE - CLOSE

def Alpha76(para_list):
    return  STD(ABS((CLOSE/DELAY(CLOSE,para_list[0])-1.0))/VOLUME,para_list[1])/MEAN(ABS((CLOSE/DELAY(CLOSE,para_list[0])-1.0))/VOLUME,para_list[1])

def Alpha77(para_list):
    return MIN(RANK(DECAYLINEAR(((((HIGH + LOW) / 2) + HIGH) - (VWAP+HIGH)),para_list[0])),RANK(DECAYLINEAR(CORR(((HIGH + LOW) / 2), MEAN(VOLUME,para_list[1]),para_list[2]),para_list[3])))

#here para_list[1] is a float
def Alpha78(para_list):
    return ((HIGH+LOW+CLOSE)/3-MEAN((HIGH+LOW+CLOSE)/3,para_list[0]))/(para_list[1]*MEAN(ABS(CLOSE-MEAN((HIGH+LOW+CLOSE)/3,para_list[0])),para_list[0]))

def Alpha79(para_list):
    return SMA(MAX(CLOSE-DELAY(CLOSE,para_list[0]),0),para_list[1],para_list[2])\
          /SMA(ABS(CLOSE-DELAY(CLOSE,para_list[0]))  ,para_list[1],para_list[2])

def Alpha80(para_list):
    return (VOLUME-DELAY(VOLUME,para_list[0]))/DELAY(VOLUME,para_list[0])

def Alpha81(para_list):
    return SMA(VOLUME,para_list[0],para_list[1])

def Alpha82(para_list):
    return SMA((TSMAX(HIGH,para_list[0])-CLOSE)/(TSMAX(HIGH,para_list[0])-TSMIN(LOW,para_list[0])),para_list[1],para_list[2])

def Alpha83(para_list):
    return (-RANK(COVIANCE(RANK(HIGH), RANK(VOLUME), para_list[0])))

def Alpha84(para_list):
    return SUM((CLOSE>DELAY(CLOSE,para_list[0]))*VOLUME+\
             (~(CLOSE>DELAY(CLOSE,para_list[0]))&(CLOSE<DELAY(CLOSE,para_list[0])))*(-VOLUME),para_list[1])

def Alpha85(para_list):
    return TSRANK((VOLUME / MEAN(VOLUME,para_list[0])),para_list[0])\
         * TSRANK((-1 * DELTA(CLOSE, para_list[1])), para_list[2])

#para_list[0] is a float
def Alpha86(para_list):
    return ( para_list[0] < (((DELAY(CLOSE, para_list[1]) - DELAY(CLOSE, para_list[2])) / para_list[2]) - ((DELAY(CLOSE, para_list[3]) - CLOSE) / para_list[3]))) *(-1.0)\
        + (~(para_list[0] < (((DELAY(CLOSE, para_list[1]) - DELAY(CLOSE, para_list[2])) / para_list[2]) - ((DELAY(CLOSE, para_list[3]) - CLOSE) / para_list[3]))))\
                       * ((((( DELAY(CLOSE, para_list[1]) - DELAY(CLOSE, para_list[2])) / para_list[2]) - ((DELAY(CLOSE, para_list[3]) - CLOSE) / para_list[3])) < 0) * 1.0\
                       + (~((((DELAY(CLOSE, para_list[1]) - DELAY(CLOSE, para_list[2])) / para_list[2]) - ((DELAY(CLOSE, para_list[3]) - CLOSE) / para_list[3])) < 0)) *(-1.0))

#LOW*0.9 + LOW*0.1 难道不就是LOW吗？改为HIGH*para_list[4] + LOW*(1-para_list[4])，因此para_list[4] is a float between 0 and 1
def Alpha87(para_list):
    return (-(RANK(DECAYLINEAR(DELTA(VWAP, para_list[0]), para_list[1]))\
          + TSRANK(DECAYLINEAR((((LOW) - VWAP) / (OPEN - ((HIGH*para_list[4] + LOW*(1-para_list[4])) / 2))), para_list[2]), para_list[3])))

def Alpha88(para_list):
    return (CLOSE-DELAY(CLOSE,para_list[0]))/DELAY(CLOSE,para_list[0])

def Alpha89(para_list):
    return (SMA(CLOSE,para_list[0],para_list[3])\
           -SMA(CLOSE,para_list[1],para_list[4])\
           -SMA(SMA(CLOSE,para_list[0],para_list[3])\
           -SMA(CLOSE,para_list[1],para_list[4]),para_list[2],para_list[5]))

def Alpha90(para_list):
    return (-RANK(CORR(RANK(VWAP), RANK(VOLUME), para_list[0])))

def Alpha91(para_list):
    return (-(RANK((CLOSE - MAX(CLOSE, para_list[0])))\
             *RANK(CORR((MEAN(VOLUME,para_list[1])), LOW, para_list[0]))))

#para_list[0] is a float between 0 and 1
def Alpha92(para_list):
    return -MAX(RANK(DECAYLINEAR(DELTA(((CLOSE* para_list[0])+ (VWAP*(1-para_list[0]))),para_list[1]),para_list[2])),\
              TSRANK(DECAYLINEAR(ABS(CORR((MEAN(VOLUME,para_list[3])), CLOSE, para_list[4])), para_list[5]), para_list[6]))

def Alpha93(para_list):
    return SUM(~(OPEN>=DELAY(OPEN,para_list[0]))*MAX((OPEN-LOW),(OPEN-DELAY(OPEN,para_list[0]))),para_list[1])

def Alpha94(para_list):
    return SUM((CLOSE>DELAY(CLOSE,para_list[0])*VOLUME\
           + (~(CLOSE>DELAY(CLOSE,para_list[0])))*(-VOLUME)*(CLOSE<DELAY(CLOSE,para_list[0]))),para_list[1])

def Alpha95(para_list):
    return STD(AMOUNT,para_list[0])

def Alpha96(para_list):
    return SMA(SMA((CLOSE-TSMIN(LOW,para_list[0]))/(TSMAX(HIGH,para_list[0])-TSMIN(LOW,para_list[0])),para_list[1],para_list[2]),para_list[3],para_list[4])

#跟Alpha95重复
def Alpha97(para_list):
    return STD(VOLUME,para_list[0])

#para_list[2] is a float
def Alpha98(para_list):
    condition = ((DELTA((SUM(CLOSE, para_list[0]) / para_list[0]), para_list[0]) / DELAY(CLOSE, para_list[0])) <= para_list[2])
    return -(condition  * ((CLOSE - TSMIN(CLOSE, para_list[0])))\
          +(~condition) * DELTA(CLOSE, para_list[1]))

def Alpha99(para_list):
    return (-RANK(COVIANCE(RANK(CLOSE), RANK(VOLUME), para_list[0])))

#跟97，95重复
def Alpha100(para_list):
    return STD(VOLUME,para_list[0])

'''just return True & False, para_list[4] is a float between 0 and 1'''
def Alpha101(para_list):
    return (-(RANK(CORR(CLOSE, SUM(MEAN(VOLUME,para_list[0]), para_list[1]), para_list[2])) < 
              RANK(CORR(RANK(((HIGH * para_list[4]) + (VWAP * (1-para_list[4])))), RANK(VOLUME), para_list[3]))))

def Alpha102(para_list):
    return SMA(MAX(VOLUME-DELAY(VOLUME,para_list[0]),0),para_list[1],para_list[2])\
          /SMA(ABS(VOLUME-DELAY(VOLUME,para_list[0]))  ,para_list[1],para_list[2])

def Alpha103(para_list):
    return ((para_list[0]-LOWDAY(LOW,para_list[0]))/para_list[0])

def Alpha104(para_list):
    return (-(DELTA(CORR(HIGH, VOLUME, para_list[0]), para_list[0]) * RANK(STD(CLOSE, para_list[1]))))

def Alpha105(para_list):
    return (-1 * CORR(RANK(OPEN), RANK(VOLUME), para_list[0]))

def Alpha106(para_list):
    return CLOSE-DELAY(CLOSE,para_list[0])

def Alpha107(para_list):
    return   -RANK(OPEN - DELAY(HIGH,  para_list[0]))\
            * RANK(OPEN - DELAY(CLOSE, para_list[0]))\
            * RANK(OPEN - DELAY(LOW,   para_list[0]))

def Alpha108(para_list):
    return (-(RANK((HIGH - MIN(HIGH, para_list[0])))**RANK(CORR((VWAP), (MEAN(VOLUME,para_list[1])), para_list[2]))))

def Alpha109(para_list):
    return SMA(HIGH-LOW,para_list[0],para_list[1])/SMA(SMA(HIGH-LOW,para_list[0],para_list[1]),para_list[0],para_list[1])

def Alpha110(para_list):
    return SUM(MAX(0,HIGH-DELAY(CLOSE,para_list[0])),para_list[1])\
          /SUM(MAX(0,-LOW+DELAY(CLOSE,para_list[0])),para_list[1])

def Alpha111(para_list):
    return SMA(VOLUME*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),para_list[0],para_list[2])\
          -SMA(VOLUME*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),para_list[1],para_list[3])

def Alpha112(para_list):
    return (SUM((CLOSE-DELAY(CLOSE,para_list[0])>0) *    (CLOSE-DELAY(CLOSE,para_list[1])),para_list[2])\
           -SUM((CLOSE-DELAY(CLOSE,para_list[0])<0) * ABS(CLOSE-DELAY(CLOSE,para_list[1])),para_list[2]))\
          /(SUM((CLOSE-DELAY(CLOSE,para_list[0])>0) *    (CLOSE-DELAY(CLOSE,para_list[1])),para_list[2])\
           +SUM((CLOSE-DELAY(CLOSE,para_list[0])<0) * ABS(CLOSE-DELAY(CLOSE,para_list[1])),para_list[2]))

def Alpha113(para_list):
    return  -(RANK(SUM(DELAY(CLOSE, para_list[0]), para_list[1]) / para_list[1]) * CORR(CLOSE, VOLUME, para_list[2]))\
            * RANK(CORR(SUM( CLOSE, para_list[0]), SUM(CLOSE, para_list[1]), para_list[2]))

def Alpha114(para_list):
    return ((RANK(DELAY(((HIGH - LOW) / (SUM(CLOSE, para_list[0]) / para_list[0])), para_list[1])) * RANK(RANK(VOLUME)))
                     / (((HIGH - LOW) / (SUM(CLOSE, para_list[0]) / para_list[0])) / (VWAP - CLOSE)))

#para_list[0] is a float between 0 and 1
def Alpha115(para_list):
    return RANK(CORR(((HIGH * para_list[0]) + (CLOSE * (1-para_list[0]))),  MEAN(VOLUME, para_list[1]),para_list[2]))\
         **RANK(CORR(TSRANK(((HIGH + LOW) / 2), para_list[3]), TSRANK(VOLUME, para_list[4]), para_list[5]))

def Alpha116(para_list):
    return REGBETA(CLOSE,SEQUENCE(para_list[0]),para_list[0])

def Alpha117(para_list):
    return ((TSRANK(VOLUME, para_list[0]) * (1 - TSRANK(((CLOSE + HIGH) - LOW), para_list[1])))* (1 - TSRANK(RET, para_list[0])))

def Alpha118(para_list):
    return SUM(HIGH-OPEN,para_list[0])/SUM(OPEN-LOW,para_list[0])

def Alpha119(para_list):
    return (RANK(DECAYLINEAR(CORR(VWAP, SUM(MEAN(VOLUME,para_list[0]), para_list[1]), para_list[2]),para_list[3]))\
           -RANK(DECAYLINEAR(TSRANK(MIN(CORR(RANK(OPEN), RANK(MEAN(VOLUME,para_list[4])), para_list[5]), para_list[6]), para_list[7]), para_list[8])))

def Alpha120(para_list):
    return (RANK((VWAP - CLOSE)) / RANK((VWAP + CLOSE)))

def Alpha121(para_list):
    return -RANK(VWAP - MIN(VWAP, para_list[0]))**TSRANK(CORR(TSRANK(VWAP, para_list[1]), TSRANK(MEAN(VOLUME,para_list[2]), para_list[3]), para_list[4]), para_list[5])

def Alpha122(para_list):
    return (SMA(SMA(SMA(np.log(CLOSE),para_list[0],para_list[1]),para_list[0],para_list[1]),para_list[0],para_list[1])\
     /DELAY(SMA(SMA(SMA(np.log(CLOSE),para_list[0],para_list[1]),para_list[0],para_list[1]),para_list[0],para_list[1]),para_list[2])) - 1.0

'''输出的是bool type'''
def Alpha123(para_list):
    return (-(RANK(CORR(SUM((HIGH + LOW) /2, para_list[0]), SUM(MEAN(VOLUME,para_list[1]), para_list[2]), para_list[3]))< RANK(CORR(LOW, VOLUME, para_list[4]))))

def Alpha124(para_list):
    return (CLOSE - VWAP) / DECAYLINEAR(RANK(TSMAX(CLOSE, para_list[0])),para_list[1])

def Alpha125(para_list):
    return (RANK(DECAYLINEAR(CORR((VWAP), MEAN(VOLUME,para_list[0]),para_list[1]), para_list[2]))\
           /RANK(DECAYLINEAR(DELTA(((CLOSE * 0.5) + (VWAP * 0.5)), para_list[3]), para_list[4])))

def Alpha126():
    return (CLOSE+HIGH+LOW)/3

#原来是平方再开方的，这里我就直接取ABS了
def Alpha127(para_list):
    return ABS(MEAN(((CLOSE-MAX(CLOSE,para_list[0]))/(MAX(CLOSE,para_list[0]))), para_list[0]))

def Alpha128(para_list):
    return 100-(100/(1+SUM(((HIGH+LOW+CLOSE)/3>DELAY((HIGH+LOW+CLOSE)/3,para_list[0]))*(HIGH+LOW+CLOSE)/3*VOLUME,para_list[1])/
                       SUM(((HIGH+LOW+CLOSE)/3<DELAY((HIGH+LOW+CLOSE)/3,para_list[0]))*(HIGH+LOW+CLOSE)/3*VOLUME,para_list[1])))

def Alpha129(para_list):
    return SUM((CLOSE-DELAY(CLOSE,para_list[0])<0) * ABS(CLOSE-DELAY(CLOSE,para_list[0])),para_list[1])


def Alpha130(para_list):
    return (RANK(DECAYLINEAR(CORR(((HIGH + LOW) / 2),MEAN(VOLUME,para_list[0]),para_list[1]),para_list[2]))\
           /RANK(DECAYLINEAR(CORR(RANK(VWAP), RANK(VOLUME), para_list[3]),para_list[4])))

def Alpha131(para_list):
    return (RANK(DELAY(VWAP, para_list[0]))**TSRANK(CORR(CLOSE,MEAN(VOLUME,para_list[1]), para_list[2]), para_list[2]))

def Alpha132(para_list):
    return MEAN(AMOUNT,para_list[0])

def Alpha133(para_list):
    return ((para_list[0]-HIGHDAY(HIGH,para_list[0]))/para_list[0])\
          -((para_list[0]-LOWDAY(LOW  ,para_list[0]))/para_list[0])

def Alpha134(para_list):
    return (CLOSE-DELAY(CLOSE,para_list[0]))/DELAY(CLOSE,para_list[0])*VOLUME

def Alpha135(para_list):
    return SMA(DELAY(CLOSE/DELAY(CLOSE,para_list[0]),para_list[1]),para_list[0],para_list[2])

def Alpha136(para_list):
    return ((-RANK(DELTA(RET, para_list[0]))) * CORR(OPEN, VOLUME, para_list[1]))

#这个就是Alpha55把最外面那层sum()去掉,那其实就相当于.rolling.sum(window=1)的情形，此处也算作是重复计算
def Alpha55(para_list):
    part_C_value = MAX(ABS(HIGH-DELAY(CLOSE,para_list[0])),\
                       ABS(LOW- DELAY(CLOSE,para_list[0])))

    part_A_value = (CLOSE+(CLOSE-OPEN)/2-DELAY(OPEN,para_list[0]))

    part_B1_cond = (ABS(HIGH-DELAY(CLOSE,para_list[0])) > ABS(LOW -DELAY(CLOSE,para_list[0])))\
                  &(ABS(HIGH-DELAY(CLOSE,para_list[0])) > ABS(HIGH-DELAY(LOW,  para_list[0])))

    part_B2_cond = (ABS(LOW- DELAY(CLOSE,para_list[0])) > ABS(HIGH-DELAY(LOW,  para_list[0])))\
                  &(ABS(LOW- DELAY(CLOSE,para_list[0])) > ABS(HIGH-DELAY(CLOSE,para_list[0])))
    
    part_B1_value=  ABS(HIGH-DELAY(CLOSE,para_list[0]))\
                  + ABS(LOW -DELAY(CLOSE,para_list[0]))/para_list[1]\
                  + ABS(DELAY(CLOSE,para_list[0])\
                       -DELAY(OPEN, para_list[0]))/para_list[2]
    '''
    part_B2pvalue=  ABS(LOW           -DELAY(CLOSE,1))\
                  + ABS(HIGH          -DELAY(CLOSE,1))/2\
                  + ABS(DELAY(CLOSE,1)-DELAY(OPEN ,1))/4 #same of the previous one
    '''
    part_B2nvalue= (ABS(HIGH-DELAY(LOW ,para_list[0])) + ABS(DELAY(CLOSE,para_list[0])-DELAY(OPEN,para_list[0]))/para_list[2])
    
    part_B_value = (part_B1_cond  | (~part_B1_cond) & part_B2_cond) * part_B1_value\
               + ((~part_B1_cond) & (~part_B2_cond))                * part_B2nvalue
    
    return part_A_value/part_B_value*part_C_value


#here para_list[0] is a float between 0 and 1
def Alpha138(para_list):
    return (-(RANK(DECAYLINEAR(DELTA((((LOW * para_list[0]) + (VWAP * (1-para_list[0])))), para_list[1]), para_list[2]))\
           -TSRANK(DECAYLINEAR(TSRANK(CORR(TSRANK(LOW, para_list[3]), TSRANK(MEAN(VOLUME,para_list[4]), para_list[5]),para_list[6]),para_list[7]),para_list[8]),para_list[9])))

def Alpha139(para_list):
    return (-CORR(OPEN, VOLUME, para_list[0]))

def Alpha140(para_list):
    return MIN(RANK(DECAYLINEAR(((RANK(OPEN) + RANK(LOW)) - (RANK(HIGH) + RANK(CLOSE))),para_list[0])),\
             TSRANK(DECAYLINEAR(CORR(TSRANK(CLOSE, para_list[1]), TSRANK(MEAN(VOLUME, para_list[2]),para_list[3]),para_list[4]),para_list[5]),para_list[5]))

def Alpha141(para_list):
    return (-RANK(CORR(RANK(HIGH), RANK(MEAN(VOLUME,para_list[0])), para_list[1])))

def Alpha142(para_list):
    return (((-RANK(TSRANK(CLOSE, para_list[0]))) * RANK(DELTA(DELTA(CLOSE,para_list[1]), para_list[1]))) * RANK(TSRANK((VOLUME/MEAN(VOLUME,para_list[2])), para_list[3])))

#Alpha143,没有定义SELF函数
def Alpha143(para_list):
    return CLOSE - CLOSE

def Alpha144(para_list):
    return SUMIF(ABS(CLOSE/DELAY(CLOSE,para_list[0])-1)/AMOUNT,para_list[1],CLOSE<DELAY(CLOSE,para_list[0]))/COUNT(CLOSE<DELAY(CLOSE,para_list[0]),para_list[1])

def Alpha145(para_list):
    return (MEAN(VOLUME,para_list[0])-MEAN(VOLUME,para_list[1]))/MEAN(VOLUME,para_list[2])

#里面有一个square我就不改了- -
def Alpha146(para_list):
    return MEAN((CLOSE-DELAY(CLOSE,para_list[0]))/DELAY(CLOSE,para_list[0])\
           -SMA((CLOSE-DELAY(CLOSE,para_list[0]))/DELAY(CLOSE,para_list[0]), para_list[1],para_list[4]),para_list[2])\
             * ((CLOSE-DELAY(CLOSE,para_list[0]))/DELAY(CLOSE,para_list[0])\
           -SMA((CLOSE-DELAY(CLOSE,para_list[0]))/DELAY(CLOSE,para_list[0]), para_list[1],para_list[4]))\
           /SMA(((CLOSE-DELAY(CLOSE,para_list[0]))/DELAY(CLOSE,para_list[0])\
               -((CLOSE-DELAY(CLOSE,para_list[0]))/DELAY(CLOSE,para_list[0])\
           -SMA(( CLOSE-DELAY(CLOSE,para_list[0]))/DELAY(CLOSE,para_list[0]),para_list[3],para_list[4])))**2,para_list[1],para_list[4])


def Alpha147(para_list):
    return  REGBETA(MEAN(CLOSE,para_list[0]), SEQUENCE(para_list[0]), para_list[0])

'''这里返回的也是个bool'''
def Alpha148(para_list):
    return -(RANK(CORR((OPEN), SUM(MEAN(VOLUME,para_list[0]), para_list[1]), para_list[2])) < RANK((OPEN - TSMIN(OPEN, para_list[3]))))

#Alpha149, BANCHMARKCLOSE没有定义，所以这个index空着
def Alpha149(para_list):
    return CLOSE - CLOSE

def Alpha150(para_list):
    return (CLOSE+HIGH+LOW)/3*VOLUME

def Alpha151(para_list):
    return  SMA(CLOSE-DELAY(CLOSE,para_list[0]),para_list[0],para_list[1])

def Alpha152(para_list):
    return SMA(MEAN(DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,para_list[0]),para_list[1]),para_list[0],para_list[1]),para_list[0]),para_list[2])\
              -MEAN(DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,para_list[0]),para_list[1]),para_list[0],para_list[1]),para_list[0]),para_list[3]),para_list[0],para_list[1])

#这里取的window是成倍数的，我不认为他们是独立的，因此我只用了一个parameter来描述
def Alpha153(para_list):
    return (MEAN(CLOSE,  para_list[0])\
           +MEAN(CLOSE,2*para_list[0])\
           +MEAN(CLOSE,4*para_list[0])\
           +MEAN(CLOSE,8*para_list[0]))/4

#这个返回的也是一个bool
def Alpha154(para_list):
    return (((VWAP - MIN(VWAP, para_list[0]))) < (CORR(VWAP, MEAN(VOLUME,para_list[1]), para_list[2])))

def Alpha155(para_list):
    return SMA(VOLUME,para_list[0],para_list[3])\
          -SMA(VOLUME,para_list[1],para_list[4])\
          -SMA(\
           SMA(VOLUME,para_list[0],para_list[3])\
          -SMA(VOLUME,para_list[1],para_list[4]),\
               para_list[2],para_list[5])

#para_list[3] is a float between 0 and 1
def Alpha156(para_list):
    return -MAX(RANK(DECAYLINEAR(DELTA(VWAP, para_list[0]), para_list[1])),\
                RANK(DECAYLINEAR((-(DELTA(((OPEN * para_list[3]) + (LOW * (1-para_list[3]))),   para_list[2])\
                                         /((OPEN * para_list[3]) + (LOW * (1-para_list[3]))))), para_list[1])))


def Alpha157(para_list):
    return (MIN(PROD(RANK(RANK(LOG(SUM(TSMIN(RANK(RANK(-RANK(DELTA((CLOSE - para_list[0]), para_list[1])))), para_list[2]), para_list[3])))), para_list[4]), para_list[5]) + TSRANK(DELAY((-RET), para_list[6]), para_list[7]))

def Alpha158(para_list):
    return  ((HIGH-SMA(CLOSE,para_list[0],para_list[1]))-(LOW-SMA(CLOSE,para_list[0],para_list[1])))/CLOSE


def Alpha159(para_list):
    return (CLOSE-SUM(MIN(LOW, DELAY(CLOSE,para_list[3])),para_list[0]))\
                 /SUM(MAX(HIGH,DELAY(CLOSE,para_list[3]))-MIN(LOW,DELAY(CLOSE,para_list[3])),para_list[0])*para_list[1]*para_list[2]\
          +(CLOSE-SUM(MIN(LOW, DELAY(CLOSE,para_list[3])),para_list[1]))\
                 /SUM(MAX(HIGH,DELAY(CLOSE,para_list[3]))-MIN(LOW,DELAY(CLOSE,para_list[3])),para_list[1])*para_list[1]*para_list[2]\
          +(CLOSE-SUM(MIN(LOW, DELAY(CLOSE,para_list[3])),para_list[2]))\
                 /SUM(MAX(HIGH,DELAY(CLOSE,para_list[3]))-MIN(LOW,DELAY(CLOSE,para_list[3])),para_list[2])*para_list[1]*para_list[2]\
                 /(para_list[0]*para_list[1]+para_list[1]*para_list[2]+para_list[2]*para_list[0])

def Alpha160(para_list):
    return SMA((CLOSE<=DELAY(CLOSE,para_list[0]))*STD(CLOSE,para_list[1]),para_list[1],para_list[2])

def Alpha161(para_list):
    return MEAN(MAX(MAX((HIGH-LOW),ABS(DELAY(CLOSE,para_list[0])-HIGH)),ABS(DELAY(CLOSE,para_list[0])-LOW)),para_list[1])

def Alpha162(para_list):
    return (SMA(MAX(CLOSE-DELAY(CLOSE,para_list[0]),0),para_list[1],para_list[2])\
           /SMA(ABS(CLOSE-DELAY(CLOSE,para_list[0]))  ,para_list[1],para_list[2])\
       -MIN(SMA(MAX(CLOSE-DELAY(CLOSE,para_list[0]),0),para_list[1],para_list[2])\
           /SMA(ABS(CLOSE-DELAY(CLOSE,para_list[0]))  ,para_list[1],para_list[2]),para_list[1]))\
              /(MAX(SMA(MAX(CLOSE-DELAY(CLOSE,para_list[0]),0),para_list[1],para_list[2])
                   /SMA(ABS(CLOSE-DELAY(CLOSE,para_list[0]))  ,para_list[1],para_list[2]),para_list[1])\
               -MIN(SMA(MAX(CLOSE-DELAY(CLOSE,para_list[0]),0),para_list[1],para_list[2])\
                   /SMA(ABS(CLOSE-DELAY(CLOSE,para_list[0]))  ,para_list[1],para_list[2]),para_list[1]))

def Alpha163(para_list):
    return RANK(((((-RET) * MEAN(VOLUME,para_list[0])) * VWAP) * (HIGH - CLOSE)))


def Alpha164(para_list):
    return SMA((((CLOSE>DELAY(CLOSE,para_list[0]))*1/(CLOSE-DELAY(CLOSE,para_list[0]))+ ~(CLOSE>DELAY(CLOSE,para_list[0]))*1)
          - MIN(((CLOSE>DELAY(CLOSE,para_list[0]))*1/(CLOSE-DELAY(CLOSE,para_list[0]))+ ~(CLOSE>DELAY(CLOSE,para_list[0]))*1),para_list[1]))/(HIGH-LOW),para_list[2],2)


def Alpha165(para_list):
    return SUMAC(CLOSE-MEAN(CLOSE,para_list[0]),para_list[0])\
         - SUMAC(CLOSE-MEAN(CLOSE,para_list[0]),para_list[0])/STD(CLOSE,para_list[0])

#**1.5保留 不然120**120估计太大了
def Alpha166(para_list):
    return -para_list[0]*((para_list[1])**1.5)*SUM((CLOSE/DELAY(CLOSE,para_list[2])-MEAN(CLOSE/DELAY(CLOSE,para_list[3])-1,para_list[4])),para_list[5])             /((20-1)*(20-2)*((SUM((CLOSE/DELAY(CLOSE,1))**2,20))**1.5))

def Alpha167(para_list):
    return SUM((CLOSE-DELAY(CLOSE,para_list[0])>0)*(CLOSE-DELAY(CLOSE,para_list[0])),para_list[1])

def Alpha168(para_list):
    return (-VOLUME/MEAN(VOLUME,para_list[0]))

def Alpha169(para_list):
    return SMA(MEAN(DELAY(SMA(CLOSE-DELAY(CLOSE,para_list[0]),para_list[1],para_list[0]),para_list[5]),para_list[2])\
              -MEAN(DELAY(SMA(CLOSE-DELAY(CLOSE,para_list[0]),para_list[1],para_list[0]),para_list[5]),para_list[3]),para_list[4],para_list[5])

def Alpha170(para_list): #rank * rank - rank almost还是rank
    return ((RANK((1 / CLOSE)) * VOLUME / MEAN(VOLUME, para_list[0]))* (HIGH * RANK(HIGH - CLOSE)) / (SUM(HIGH, para_list[1]) / para_list[1])) - RANK(VWAP - DELAY(VWAP, para_list[1]))

def Alpha171(para_list):
    return  -((LOW - CLOSE) * (OPEN**para_list[0])) / ((CLOSE - HIGH) * (CLOSE**para_list[0]))

def Alpha172(para_list):
    return MEAN(ABS(SUM((LD>0 & LD>HD)*LD,para_list[0])/SUM(TR,para_list[1])\
                   -SUM((HD>0 & HD>LD)*HD,para_list[0])/SUM(TR,para_list[1]))\
                  /(SUM((LD>0 & LD>HD)*LD,para_list[0])/SUM(TR,para_list[1])\
                   +SUM((HD>0 & HD>LD)*HD,para_list[0])/SUM(TR,para_list[1])),para_list[2])

#3-2+1或许是某种玄学，没改
def Alpha173(para_list):
    return 3*SMA(CLOSE,para_list[0],para_list[1])\
          -2*SMA(SMA(CLOSE,para_list[0],para_list[1]),para_list[0],para_list[1])\
            +SMA(SMA(SMA(np.log(CLOSE),para_list[0],para_list[1]),para_list[0],para_list[1]),para_list[0],para_list[1])

def Alpha174(para_list):
    return SMA((CLOSE>DELAY(CLOSE,para_list[0]))*STD(CLOSE,para_list[1]),para_list[1],para_list[2])

def Alpha175(para_list):
    return MEAN(MAX(MAX((HIGH-LOW),ABS(DELAY(CLOSE,para_list[0])-HIGH)),ABS(DELAY(CLOSE,para_list[0])-LOW)),para_list[1])

def Alpha176(para_list):
    return CORR(RANK((CLOSE - TSMIN(LOW, para_list[0])) / (TSMAX(HIGH, para_list[0]) - TSMIN(LOW,para_list[0]))), RANK(VOLUME), para_list[1])

def Alpha177(para_list):
    return ((para_list[0]-HIGHDAY(HIGH,para_list[0]))/para_list[0])

def Alpha178(para_list):
    return (CLOSE-DELAY(CLOSE,para_list[0]))/DELAY(CLOSE,para_list[0])*VOLUME

def Alpha179(para_list):
    return (RANK(CORR(VWAP, VOLUME, para_list[0])) * RANK(CORR(RANK(LOW), RANK(MEAN(VOLUME,para_list[1])), para_list[2])))

def Alpha180(para_list):
    return  (MEAN(VOLUME,para_list[0]) < VOLUME) * (-TSRANK(ABS(DELTA(CLOSE, para_list[1])), para_list[2])) * SIGN(DELTA(CLOSE, para_list[1]))\
         + ~(MEAN(VOLUME,para_list[0]) < VOLUME) * (-VOLUME)

#Alpha181 drop for the BENCHMARKINDEX
def Alpha181(para_list):
    return CLOSE - CLOSE

#Alpha182 drop for the BENCHMARKINDEX
def Alpha182(para_list):
    return CLOSE - CLOSE

def Alpha183(para_list):
    return MAX(SUMAC(CLOSE-MEAN(CLOSE,para_list[0]),para_list[0]),para_list[0])\
          -MIN(SUMAC(CLOSE-MEAN(CLOSE,para_list[0]),para_list[0]),para_list[0])/STD(CLOSE,para_list[0])

def Alpha184(para_list):
    return (RANK(CORR(DELAY((OPEN - CLOSE), para_list[0]), CLOSE, para_list[1])) + RANK((OPEN - CLOSE)))

#**2也没动
def Alpha185(para_list):
    return RANK((-((1 - (OPEN / CLOSE))**2)))

def Alpha186(para_list):
    return (MEAN(ABS(SUM(((LD>0) & (LD>HD))*LD,para_list[0])/SUM(TR,para_list[0])\
                    -SUM(((HD>0) & (HD>LD))*HD,para_list[0])/SUM(TR,para_list[0]))\
                   /(SUM(((LD>0) & (LD>HD))*LD,para_list[0])/SUM(TR,para_list[0])\
                    +SUM(((HD>0) & (HD>LD))*HD,para_list[0])/SUM(TR,para_list[0])),para_list[1])\
     +DELAY(MEAN(ABS(SUM(((LD>0) & (LD>HD))*LD,para_list[0])/SUM(TR,para_list[0])\
                    -SUM(((HD>0) & (HD>LD))*HD,para_list[0])/SUM(TR,para_list[0]))\
                   /(SUM(((LD>0) & (LD>HD))*LD,para_list[0])/SUM(TR,para_list[0])\
                    +SUM(((HD>0) & (HD>LD))*HD,para_list[0])/SUM(TR,para_list[0])),para_list[1]),para_list[1]))/2

def Alpha187(para_list):
    return SUM(~(OPEN<=DELAY(OPEN,para_list[0])) * MAX((HIGH-OPEN),(OPEN-DELAY(OPEN,para_list[0]))),para_list[1])

def Alpha188(para_list):
    return ((HIGH-LOW-SMA(HIGH-LOW,para_list[0],2))/SMA(HIGH-LOW,para_list[0],2))

def Alpha189(para_list):
    return MEAN(ABS(CLOSE-MEAN(CLOSE,para_list[0])),para_list[0])

''' Alpha190我很无奈。。。
def Alpha190(para_list):
    return  
LOG((COUNT(CLOSE/DELAY(CLOSE)-1>((CLOSE/DELAY(CLOSE,19))^(1/20)-1),20)-1)\
 *(SUMIF(((CLOSE/DELAY(CLOSE)-1-(CLOSE/DELAY(CLOSE,19))^(1/20)-1))^2,20,\
           CLOSE/DELAY(CLOSE)-1<(CLOSE/DELAY(CLOSE,19))^(1/20)1))\
 /((COUNT((CLOSE/DELAY(CLOSE)-1<(CLOSE/DELAY(CLOSE,19))^(1/20)-1),20))\
 *(SUMIF(( CLOSE/DELAY(CLOSE)-1-((CLOSE/DELAY(CLOSE,19))^(1/20)-1))^2,20,\
           CLOSE/DELAY(CLOSE)-1>(CLOSE/DELAY(CLOSE,19))^(1/20)-1))) )
'''

def Alpha191(para_list):
    return ((CORR(MEAN(VOLUME,para_list[0]), LOW, para_list[1]) + ((HIGH + LOW) / 2)) - CLOSE)

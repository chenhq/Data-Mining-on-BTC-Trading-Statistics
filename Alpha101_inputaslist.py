#IndClass.industry and IndNeutralize没有实现，因此只有82的index

def Scale(A, n=1):
    return A.rolling(n).apply(lambda x: x/x.sum()).fillna(method = 'ffill').fillna(method='bfill')

#Average Daily Volume
def ADV(n):
    return VOLUME.rolling(window = n, min_periods = 1).mean()



#para_list[3] is a float, initial value = 0.5
def ALPHA1(para_list):
    return(RANK(TS_ARGMAX(SIGN_POWER(((RET<0)*STD(RET,para_list[0])+(~(RET<0))*CLOSE),para_list[1]),para_list[2]))-para_list[3]/100)

def ALPHA2(para_list):
    return (-CORR(RANK(DELTA(LOG(VOLUME),para_list[0])),RANK(((CLOSE-OPEN)/OPEN)),para_list[1]))

def ALPHA3(para_list):
    return (-CORR(RANK(OPEN),RANK(VOLUME),para_list[0]))

def ALPHA4(para_list):
    return (-TSRANK(RANK(LOW),para_list[0]))

def ALPHA5(para_list):
    (RANK((OPEN-(SUM(VWAP,para_list[0])/para_list[0])))*(-ABS(RANK((CLOSE-VWAP)))))

def ALPHA6(para_list):
    return (-CORR(OPEN,VOLUME,para_list[0]))

def ALPHA7(para_list):
    return ((ADV(para_list[0])<VOLUME)*((-TSRANK(ABS(DELTA(CLOSE,para_list[1])),para_list[2]))*SIGN(DELTA(CLOSE,para_list[3])))+(~(ADV(para_list[0])<VOLUME))*(-1))

def ALPHA8(para_list):
    return (-RANK(((SUM(OPEN,para_list[0])*SUM(RET,para_list[0]))\
            -DELAY((SUM(OPEN,para_list[0])*SUM(RET,para_list[0])),para_list[1]))))

def ALPHA9(para_list):
    return ((0<TSMIN(DELTA(CLOSE,para_list[0]),para_list[1]))* DELTA(CLOSE,para_list[0])\
        +  ~(0<TSMIN(DELTA(CLOSE,para_list[0]),para_list[1]))\
            *((TSMAX(DELTA(CLOSE,para_list[0]),para_list[1])<0)*DELTA(CLOSE,para_list[0])\
           -(~(TSMAX(DELTA(CLOSE,para_list[0]),para_list[1])<0)*DELTA(CLOSE,para_list[0]))))

def ALPHA10(para_list):
    return RANK(((0<TSMIN(DELTA(CLOSE,para_list[0]),para_list[1]))*DELTA(CLOSE,para_list[0])\
                +~(0<TSMIN(DELTA(CLOSE,para_list[0]),para_list[1]))*((TSMAX(DELTA(CLOSE,para_list[0]),para_list[1])<0)*DELTA(CLOSE,para_list[0])+~(TSMAX(DELTA(CLOSE,para_list[0]),para_list[1])<0)*(-1*DELTA(CLOSE,para_list[0])))))

def ALPHA11(para_list):
    return ((RANK(TSMAX((VWAP-CLOSE),para_list[0]))+RANK(TSMIN((VWAP-CLOSE),para_list[0])))*RANK(DELTA(VOLUME,para_list[0])))

def ALPHA12(para_list):
    return ((DELTA(VOLUME,para_list[0]))*(-1*DELTA(CLOSE,para_list[0])))

def ALPHA13(para_list):
    return (-1*RANK(COVIANCE(RANK(CLOSE),RANK(VOLUME),para_list[0])))

def ALPHA14(para_list):
    return ((-1*RANK(DELTA(RET,para_list[0])))*CORR(OPEN,VOLUME,para_list[1]))

def ALPHA15(para_list):
    return (-1*SUM(RANK(CORR(RANK(HIGH),RANK(VOLUME),para_list[0])),para_list[1]))

def ALPHA16(para_list):
    return (-1*RANK(COVIANCE(RANK(HIGH),RANK(VOLUME),para_list[0])))

def ALPHA17(para_list):
    return(((-1*RANK(TSRANK(CLOSE,para_list[0])))*RANK(DELTA(DELTA(CLOSE,para_list[1]),para_list[1])))*RANK(TSRANK((VOLUME/ADV(para_list[2])),para_list[3])))
    
def ALPHA18(para_list):
    return (-1*RANK(((STD(ABS((CLOSE-OPEN)),para_list[0])+(CLOSE-OPEN))+CORR(CLOSE,OPEN,para_list[1]))))

#para_list[2] initial value is 250
def ALPHA19(para_list):
    return ((-1*SIGN(((CLOSE-DELAY(CLOSE,para_list[0]))+DELTA(CLOSE,para_list[1]))))*(1+RANK((1+SUM(RET,para_list[2])))))

def ALPHA20(para_list):
    return (((-1*RANK((OPEN-DELAY(HIGH,para_list[0]))))*RANK((OPEN-DELAY(CLOSE,para_list[0]))))*RANK((OPEN-DELAY(LOW,para_list[0]))))

def ALPHA21(para_list):
    return ((((SUM(CLOSE,para_list[0])/para_list[0])+STD(CLOSE,para_list[0]))<(SUM(CLOSE,para_list[1])/para_list[1]))*(-1*1)\
           +~(((SUM(CLOSE,para_list[0])/para_list[0])+STD(CLOSE,para_list[0]))<(SUM(CLOSE,para_list[1])/para_list[1]))\
            *(((SUM(CLOSE,para_list[1])/para_list[1])<((SUM(CLOSE,para_list[0])/para_list[0])-STD(CLOSE,para_list[0])))*1\
           +~((SUM(CLOSE,para_list[1])/para_list[1])<((SUM(CLOSE,para_list[0])/para_list[0])-STD(CLOSE,para_list[0])))*(((1<(VOLUME/ADV(para_list[2]))|((VOLUME/ADV(para_list[2]))==1))*1-~(1<(VOLUME/ADV(para_list[2]))|((VOLUME/ADV(para_list[2]))==1))*1))))

def ALPHA22(para_list):
    return (-1*(DELTA(CORR(HIGH,VOLUME,para_list[0]),para_list[0])*RANK(STD(CLOSE,para_list[1]))))

def ALPHA23(para_list):
    return (((SUM(HIGH,para_list[0])/para_list[0])<HIGH)*(-1*DELTA(HIGH,para_list[1])))

def ALPHA24(para_list):
    return (((DELTA((SUM(CLOSE,para_list[0])/para_list[0]),para_list[0])/DELAY(CLOSE,para_list[0]))<= para_list[1]/100)*(-1*(CLOSE-TSMIN(CLOSE,para_list[0])))+ ((DELTA((SUM(CLOSE,para_list[0])/para_list[0]),para_list[0])/DELAY(CLOSE,para_list[0])) > para_list[1]/100)*(-1*DELTA(CLOSE,para_list[2])))

def ALPHA25(para_list):
    return RANK(((((-1*RET)*ADV(para_list[0]))*VWAP)*(HIGH-CLOSE)))

def ALPHA26(para_list):
    return (-1*TSMAX(CORR(TSRANK(VOLUME,para_list[0]),TSRANK(HIGH,para_list[1]),para_list[2]),para_list[3]))

def ALPHA27(para_list):
    return ((para_list[2]/100<RANK((SUM(CORR(RANK(VOLUME),RANK(VWAP),para_list[0]),para_list[1])/2.0)))*(-1*1)+~(para_list[2]/100<RANK((SUM(CORR(RANK(VOLUME),RANK(VWAP),para_list[0]),para_list[1])/2.0)))*1)

def ALPHA28(para_list):
    return SCALE(((CORR(ADV(para_list[0]),LOW,para_list[1])+((HIGH+LOW)/2))-CLOSE))

def ALPHA29(para_list):
    return MIN(PROD(RANK(RANK(SCALE(LOG(SUM(TSMIN(RANK(RANK((-1*RANK(DELTA((CLOSE-para_list[0]),para_list[1]))))),para_list[2]),para_list[3]))))),para_list[4]),para_list[5])+TSRANK(DELAY((-1*RET),para_list[6]),para_list[7])

def ALPHA30(para_list):
    return (((1.0-RANK(((SIGN((CLOSE-DELAY(CLOSE,para_list[0])))+SIGN((DELAY(CLOSE,para_list[0])-DELAY(CLOSE,para_list[1]))))+SIGN((DELAY(CLOSE,para_list[1])-DELAY(CLOSE,para_list[2]))))))*SUM(VOLUME,para_list[3]))/SUM(VOLUME,para_list[4]))

def ALPHA31(para_list):
    return ((RANK(RANK(RANK(DECAYLINEAR((-1*RANK(RANK(DELTA(CLOSE,para_list[0])))),para_list[0]))))+RANK((-1*DELTA(CLOSE,para_list[1]))))+SIGN(SCALE(CORR(ADV(para_list[2]),LOW,para_list[3]))))

def ALPHA32(para_list):
    return (SCALE(((SUM(CLOSE,para_list[0])/para_list[0])-CLOSE))+(para_list[1]*SCALE(CORR(VWAP,DELAY(CLOSE,para_list[2]),para_list[3]))))

def ALPHA33(para_list):
    return RANK((-1*((1-(OPEN/CLOSE))**(para_list[0]/100))))

def ALPHA34(para_list):
    return RANK(((1-RANK((STD(RET,para_list[0])/STD(RET,para_list[1]))))+(1-RANK(DELTA(CLOSE,para_list[2])))))

def ALPHA35(para_list):
    return ((TSRANK(VOLUME,para_list[0])*(1-TSRANK(((CLOSE+HIGH)-LOW),para_list[1])))*(1-TSRANK(RET,para_list[2])))

#para_list[0] is 2.21, para_list[3]=0.7, para_list[4]=0.73, para_list[9] = 0.6
def ALPHA36(para_list):
    return (((((para_list[0]/30*RANK(CORR((CLOSE-OPEN),DELAY(VOLUME,para_list[1]),para_list[2])))+(para_list[3]/70*RANK((OPEN-CLOSE))))+(para_list[4]/70*RANK(TSRANK(DELAY((-1*RET),para_list[5]),para_list[6]))))+RANK(ABS(CORR(VWAP,ADV(para_list[7]),para_list[8]))))+(para_list[9]/100*RANK((((SUM(CLOSE,para_list[10])/para_list[10])-OPEN)*(CLOSE-OPEN)))))

def ALPHA37(para_list):
    return (RANK(CORR(DELAY((OPEN-CLOSE),para_list[0]),CLOSE,para_list[1]))+RANK((OPEN-CLOSE)))

def ALPHA38(para_list):
    return ((-1*RANK(TSRANK(CLOSE,para_list[0])))*RANK((CLOSE/OPEN)))

def ALPHA39(para_list):
    return ((-1*RANK((DELTA(CLOSE,para_list[0])*(1-RANK(DECAYLINEAR((VOLUME/ADV(para_list[1])),para_list[2]))))))*(1+RANK(SUM(RET,para_list[3]))))

def ALPHA40(para_list):
    return ((-1*RANK(STD(HIGH,para_list[0])))*CORR(HIGH,VOLUME,para_list[0]))

#para_list[0] = 0.5
def ALPHA41(para_list):
    return (((HIGH*LOW)**(para_list[0]/100))-VWAP)

def ALPHA42(para_list):
    return (RANK((VWAP-CLOSE))/RANK((VWAP+CLOSE)))

def ALPHA43(para_list):
    return (TSRANK((VOLUME/ADV(para_list[0])),para_list[0])*TSRANK((-1*DELTA(CLOSE,para_list[1])),para_list[2]))

def ALPHA44(para_list):
    return (-1*CORR(HIGH,RANK(VOLUME),para_list[0]))

def ALPHA45(para_list):
    return (-1*((RANK((SUM(DELAY(CLOSE,para_list[0]),para_list[1])/para_list[1]))*CORR(CLOSE,VOLUME,para_list[2]))*RANK(CORR(SUM(CLOSE,para_list[3]),SUM(CLOSE,para_list[4]),para_list[5]))))

#para_list[0] = 0.25, 
def ALPHA46(para_list):
    return ((para_list[0]/100<(((DELAY(CLOSE,para_list[1])-DELAY(CLOSE,para_list[2]))/para_list[2])-((DELAY(CLOSE,para_list[1])-CLOSE)/para_list[1])))*(-1*1)\
        +~(para_list[0]/100<(((DELAY(CLOSE,para_list[1])-DELAY(CLOSE,para_list[2]))/para_list[2])-((DELAY(CLOSE,para_list[1])-CLOSE)/para_list[1])))*(((((DELAY(CLOSE,para_list[2])-DELAY(CLOSE,para_list[1]))/para_list[1])-((DELAY(CLOSE,para_list[1])-CLOSE)/para_list[1]))<0)*1+((((DELAY(CLOSE,para_list[2])-DELAY(CLOSE,para_list[1]))/para_list[1])-((DELAY(CLOSE,para_list[1])-CLOSE)/para_list[1]))>=0)*((-1*1)*(CLOSE-DELAY(CLOSE,para_list[3])))))

def ALPHA47(para_list):
    return ((((RANK((1/CLOSE))*VOLUME)/ADV(para_list[0]))*((HIGH*RANK((HIGH-CLOSE)))/(SUM(HIGH,para_list[1])/para_list[1])))-RANK((VWAP-DELAY(VWAP,para_list[2]))))

def ALPHA48(para_list):
    return (((((DELAY(CLOSE,para_list[0])-DELAY(CLOSE,para_list[1]))/para_list[1])-((DELAY(CLOSE,para_list[1])-CLOSE)/para_list[1]))<(-para_list[2]/100))*1\
        +((((DELAY(CLOSE,para_list[0])-DELAY(CLOSE,para_list[1]))/para_list[1])-((DELAY(CLOSE,para_list[1])-CLOSE)/para_list[1]))>(-para_list[2]/100))*((-1*1)*(CLOSE-DELAY(CLOSE,para_list[3]))))

def ALPHA49(para_list):
    return (-1*TSMAX(RANK(CORR(RANK(VOLUME),RANK(VWAP),para_list[0])),para_list[1]))

def ALPHA50(para_list):
    return (((((DELAY(CLOSE,para_list[0])-DELAY(CLOSE,para_list[1]))/para_list[1])-((DELAY(CLOSE,para_list[1])-CLOSE)/para_list[1]))<(-para_list[2]/100))*1\
        +((((DELAY(CLOSE,para_list[0])-DELAY(CLOSE,para_list[1]))/para_list[1])-((DELAY(CLOSE,para_list[1])-CLOSE)/para_list[1]))>(-para_list[2]/100))*((-1*1)*(CLOSE-DELAY(CLOSE,para_list[3]))))

def ALPHA51(para_list):
    return ((((-1*TSMIN(LOW,para_list[0]))+DELAY(TSMIN(LOW,para_list[0]),para_list[0]))*RANK(((SUM(RET,para_list[1])-SUM(RET,para_list[2]))/para_list[3])))*TSRANK(VOLUME,para_list[4]))

def ALPHA52(para_list):
    return (-1*DELTA((((CLOSE-LOW)-(HIGH-CLOSE))/(CLOSE-LOW)),para_list[0]))

#para_list[0] = 5, 
def ALPHA53(para_list):
    return ((-1*((LOW-CLOSE)*(OPEN**(para_list[0]/20))))/((LOW-HIGH)*(CLOSE**(para_list[0]/20))))

def ALPHA54(para_list):
    return (-1*CORR(RANK(((CLOSE-TSMIN(LOW,para_list[0]))/(TSMAX(HIGH,para_list[1])-TSMIN(LOW,para_list[2])))),RANK(VOLUME),para_list[3]))

# cap表示market cap， 不适用于BTC
def ALPHA55(para_list):
    return CLOSE - CLOSE + 1#(0-(1*(RANK((SUM(RET,10)/SUM(SUM(RET,2),3)))*RANK((RET*CAP)))))

def ALPHA56(para_list):
    return (0-(1*((CLOSE-VWAP)/DECAYLINEAR(RANK(TS_ARGMAX(CLOSE,para_list[0])),para_list[1]))))

def ALPHA57(para_list):
    return (0-(para_list[0]*((para_list[1]*SCALE(RANK(((((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW))*VOLUME))))-SCALE(RANK(TS_ARGMAX(CLOSE,para_list[2]))))))

#para_list[0] = 16.1219， para_list[2] = 17.9282
def ALPHA58(para_list):
    return (RANK((VWAP-TSMIN(VWAP,para_list[0]/6)))<RANK(CORR(VWAP,ADV(para_list[1]),para_list[2])))

def ALPHA59(para_list):
    return ((RANK(CORR(VWAP,SUM(ADV(para_list[0]),para_list[1]),para_list[2]))<RANK(((RANK(OPEN)+RANK(OPEN))<(RANK(((HIGH+LOW)/2))+RANK(HIGH)))))*-1)

def ALPHA60(para_list):
    return ((RANK(CORR(SUM(((OPEN*para_list[0]/100)+(LOW*(1-para_list[0]/100))),para_list[0]),SUM(ADV(para_list[2]),para_list[1]),para_list[3]))<RANK(DELTA(((((HIGH+LOW)/2)*para_list[0]/100)+(VWAP*(1-para_list[0]/100))),para_list[4])))*-1)

#para_list[4] is 13.635
def ALPHA61(para_list):
    return ((RANK(CORR(((OPEN*para_list[0]/100)+(VWAP*(1-para_list[0]/100))),SUM(ADV(para_list[1]),para_list[2]),para_list[3]))<RANK((OPEN-TSMIN(OPEN,para_list[4]/10))))*-1)

def ALPHA62(para_list):
    return ((RANK(DECAYLINEAR(DELTA(VWAP,para_list[0]),para_list[1]))+TSRANK(DECAYLINEAR(((((LOW*para_list[2]/100)+(LOW*(1-para_list[2]/100)))-VWAP)/(OPEN-((HIGH+LOW)/2))),para_list[3]),para_list[4]))*-1)

def ALPHA63(para_list):
    return ((TSRANK(CORR(RANK(HIGH),RANK(ADV(para_list[0])),para_list[1]),para_list[2])<RANK(DELTA(((CLOSE*para_list[3]/100)+(LOW*(1-para_list[3]/100))),para_list[4])))*-1)

def ALPHA64(para_list):
    return MAX(TSRANK(DECAYLINEAR(CORR(TSRANK(CLOSE,para_list[0]),TSRANK(ADV(para_list[1]),para_list[2]),para_list[3]),para_list[4]),para_list[5]),TSRANK(DECAYLINEAR((RANK(((LOW+OPEN)-(VWAP+VWAP)))**(para_list[6]/50)),para_list[7]),para_list[8]))

def ALPHA65(para_list):
    return (RANK(DECAYLINEAR(CORR(((HIGH+LOW)/2),ADV(para_list[0]),para_list[1]),para_list[2]))/RANK(DECAYLINEAR(CORR(TSRANK(VWAP,para_list[3]),TSRANK(VOLUME,para_list[4]),para_list[5]),para_list[6])))

def ALPHA66(para_list):
    return (MAX(RANK(DECAYLINEAR(DELTA(VWAP,para_list[0]),para_list[1])),TSRANK(DECAYLINEAR(((DELTA(((OPEN*para_list[2]/100)+(LOW*(1-para_list[2]/100))),para_list[3])/((OPEN*para_list[4]/100)+(LOW*(1-para_list[4]/100))))*-1),para_list[5]),para_list[6]))*-1)

def ALPHA67(para_list):
    return ((RANK(CORR(CLOSE,SUM(ADV(para_list[0]),para_list[1]),para_list[2]))<RANK(CORR(RANK(((HIGH*para_list[3]/100)+(VWAP*(1-para_list[3]/100)))),RANK(VOLUME),para_list[4])))*-1)

def ALPHA68(para_list):
    return (RANK(CORR(VWAP,VOLUME,para_list[0]))<RANK(CORR(RANK(LOW),RANK(ADV(para_list[2])),para_list[1])))

def ALPHA69(para_list):
    return MIN(RANK(DECAYLINEAR(((((HIGH+LOW)/2)+HIGH)-(VWAP+HIGH)),para_list[0])),RANK(DECAYLINEAR(CORR(((HIGH+LOW)/2),ADV(para_list[1]),para_list[2]),para_list[3])))

def ALPHA70(para_list):
    return (RANK(CORR(SUM(((LOW*para_list[0]/100)+(VWAP*(1-para_list[0]/100))),para_list[1]),SUM(ADV(para_list[2]),para_list[3]),para_list[4]))**RANK(CORR(RANK(VWAP),RANK(VOLUME),para_list[5])))

def ALPHA71(para_list):
    return ((RANK(DELAY(((HIGH-LOW)/(SUM(CLOSE,para_list[0])/para_list[0])),para_list[1]))*RANK(RANK(VOLUME)))/(((HIGH-LOW)/(SUM(CLOSE,para_list[2])/para_list[2]))/(VWAP-CLOSE)))

def ALPHA72(para_list):
    return SIGN_POWER(TSRANK((VWAP-TSMAX(VWAP,para_list[0])),para_list[1]),DELTA(CLOSE,para_list[2]))

def ALPHA73(para_list):
    return (RANK(CORR(((HIGH*para_list[0]/100)+(CLOSE*(1-para_list[0]/100))),ADV(para_list[1]),para_list[2]))**RANK(CORR(TSRANK(((HIGH+LOW)/2),para_list[3]),TSRANK(VOLUME,para_list[4]),para_list[5])))

def ALPHA74(para_list):
    return ((TSRANK(CORR(CLOSE,SUM(ADV(para_list[0]),para_list[1]),para_list[2]),para_list[3])<RANK(((OPEN+CLOSE)-(VWAP+OPEN))))*-1)

def ALPHA75(para_list):
    return MIN(RANK(DECAYLINEAR(((RANK(OPEN)+RANK(LOW))-(RANK(HIGH)+RANK(CLOSE))),para_list[0])),TSRANK(DECAYLINEAR(CORR(TSRANK(CLOSE,para_list[1]),TSRANK(ADV(para_list[2]),para_list[3]),para_list[4]),para_list[5]),para_list[6]))

def ALPHA76(para_list):
    return MIN(TSRANK(DECAYLINEAR(((((HIGH+LOW)/2)+CLOSE)<(LOW+OPEN)),para_list[0]),para_list[1]),TSRANK(DECAYLINEAR(CORR(RANK(LOW),RANK(ADV(para_list[3])),para_list[2]),para_list[4]),para_list[5]))

def ALPHA77(para_list):
    return ((RANK((VWAP-TSMIN(VWAP,para_list[0])))**TSRANK(CORR(TSRANK(VWAP,para_list[1]),TSRANK(ADV(para_list[2]),para_list[3]),para_list[4]),para_list[5]))*-1)

def ALPHA78(para_list):
    return (RANK((OPEN-TSMIN(OPEN,para_list[0])))<TSRANK((RANK(CORR(SUM(((HIGH+LOW)/2),para_list[1]),SUM(ADV(para_list[2]),para_list[3]),para_list[4]))**(para_list[5]/20)),para_list[6]))

def ALPHA79(para_list):
    return (MAX(TSRANK(DECAYLINEAR(CORR(RANK(VWAP),RANK(VOLUME),para_list[0]),para_list[1]),para_list[2]),TSRANK(DECAYLINEAR(TS_ARGMAX(CORR(TSRANK(CLOSE,para_list[3]),TSRANK(ADV(para_list[4]),para_list[5]),para_list[6]),para_list[7]),para_list[8]),para_list[9]))*-1)

def ALPHA80(para_list):
    return (RANK(DECAYLINEAR(CORR(VWAP,SUM(ADV(para_list[0]),para_list[1]),para_list[2]),para_list[3]))-RANK(DECAYLINEAR(TSRANK(TS_ARGMIN(CORR(RANK(OPEN),RANK(ADV(para_list[4])),para_list[5]),para_list[6]),para_list[7]),para_list[8])))

def ALPHA81(para_list):
    return ((RANK(CORR(SUM(((HIGH+LOW)/2),para_list[0]),SUM(ADV(para_list[1]),para_list[2]),para_list[3]))<RANK(CORR(LOW,VOLUME,para_list[4])))*-1)

def ALPHA82(para_list):
    return ((CLOSE-OPEN)/((HIGH-LOW)+para_list[0]/1000))



Alpha_para_num = {}
Alpha_para_num['ALPHA1'] = 4
Alpha_para_num['ALPHA2'] = 2
Alpha_para_num['ALPHA3'] = 1
Alpha_para_num['ALPHA4'] = 1
Alpha_para_num['ALPHA5'] = 1
Alpha_para_num['ALPHA6'] = 1
Alpha_para_num['ALPHA7'] = 3
Alpha_para_num['ALPHA8'] = 2
Alpha_para_num['ALPHA9'] = 2
Alpha_para_num['ALPHA10'] = 2
Alpha_para_num['ALPHA11'] = 1
Alpha_para_num['ALPHA12'] = 1
Alpha_para_num['ALPHA13'] = 1
Alpha_para_num['ALPHA14'] = 2
Alpha_para_num['ALPHA15'] = 2
Alpha_para_num['ALPHA16'] = 1
Alpha_para_num['ALPHA17'] = 4
Alpha_para_num['ALPHA18'] = 2
Alpha_para_num['ALPHA19'] = 3
Alpha_para_num['ALPHA20'] = 1
Alpha_para_num['ALPHA21'] = 3
Alpha_para_num['ALPHA22'] = 2
Alpha_para_num['ALPHA23'] = 2
Alpha_para_num['ALPHA24'] = 3
Alpha_para_num['ALPHA25'] = 1
Alpha_para_num['ALPHA26'] = 4
Alpha_para_num['ALPHA27'] = 3
Alpha_para_num['ALPHA28'] = 1
Alpha_para_num['ALPHA29'] = 8
Alpha_para_num['ALPHA30'] = 5
Alpha_para_num['ALPHA31'] = 4
Alpha_para_num['ALPHA32'] = 4
Alpha_para_num['ALPHA33'] = 1
Alpha_para_num['ALPHA34'] = 3
Alpha_para_num['ALPHA35'] = 3
Alpha_para_num['ALPHA36'] = 11
Alpha_para_num['ALPHA37'] = 2
Alpha_para_num['ALPHA38'] = 1
Alpha_para_num['ALPHA39'] = 4
Alpha_para_num['ALPHA40'] = 1
Alpha_para_num['ALPHA41'] = 1
Alpha_para_num['ALPHA42'] = 0
Alpha_para_num['ALPHA43'] = 3
Alpha_para_num['ALPHA44'] = 1
Alpha_para_num['ALPHA45'] = 6
Alpha_para_num['ALPHA46'] = 4
Alpha_para_num['ALPHA47'] = 3
Alpha_para_num['ALPHA48'] = 4
Alpha_para_num['ALPHA49'] = 2
Alpha_para_num['ALPHA50'] = 4
Alpha_para_num['ALPHA51'] = 5
Alpha_para_num['ALPHA52'] = 1
Alpha_para_num['ALPHA53'] = 1
Alpha_para_num['ALPHA54'] = 4
Alpha_para_num['ALPHA55'] = 0
Alpha_para_num['ALPHA56'] = 2
Alpha_para_num['ALPHA57'] = 3
Alpha_para_num['ALPHA58'] = 3
Alpha_para_num['ALPHA59'] = 3
Alpha_para_num['ALPHA60'] = 5
Alpha_para_num['ALPHA61'] = 5
Alpha_para_num['ALPHA62'] = 5
Alpha_para_num['ALPHA63'] = 5
Alpha_para_num['ALPHA64'] = 9
Alpha_para_num['ALPHA65'] = 7
Alpha_para_num['ALPHA66'] = 7
Alpha_para_num['ALPHA67'] = 5
Alpha_para_num['ALPHA68'] = 3
Alpha_para_num['ALPHA69'] = 4
Alpha_para_num['ALPHA70'] = 6
Alpha_para_num['ALPHA71'] = 3
Alpha_para_num['ALPHA72'] = 3
Alpha_para_num['ALPHA73'] = 6
Alpha_para_num['ALPHA74'] = 4
Alpha_para_num['ALPHA75'] = 7
Alpha_para_num['ALPHA76'] = 6
Alpha_para_num['ALPHA77'] = 6
Alpha_para_num['ALPHA78'] = 7
Alpha_para_num['ALPHA79'] = 10
Alpha_para_num['ALPHA80'] = 9
Alpha_para_num['ALPHA81'] = 5
Alpha_para_num['ALPHA82'] = 1
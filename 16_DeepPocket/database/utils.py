import talib as tb
from math import sqrt,isnan
import numpy as np
import warnings
warnings.filterwarnings("error")

def calculate_csi(high, low, close,atr, timeperiod = 14):
    adx = tb.ADX(high, low, close, timeperiod=timeperiod)
    csi = []
    M = 5000
    C = 0 # ETORO ?? PROVJERITI JOS 
    V = 50
    K =  100 * ((V / sqrt(M) / (150 + C)))
    for i in range(len(high)):
        value = np.nan
        if i - timeperiod >= 0:
            value = adx[i]*atr[i-timeperiod]  * K

        csi.append(value)

    csi_norm = [csi[i]/csi[i-1] if i >0 else 0.0 for i in range(len(csi))]

    return csi_norm

def calculate_demand_index(open,high, low, close, volume, timeperiod = 14):
    h = [max(high[i-2:i+1]) for i in range(2,len(high))]
    l = [min(low[i-2:i+1]) for i in range(2,len(low))]
    va = np.array(h) - np.array(l)
    
    va = tb.SMA(va, timeperiod)
    demand_index = [np.nan for i in range(timeperiod)]
    for i in range(timeperiod,len(close)):
        if open[i] == 0:
            open[i] = 0.0001
        p = (close[i] - open[i])/ open[i]
        di = 0.0
        vm = va[i-2]
        bp = 1.0
        sp = 1.0
        v = volume[i]
   
        if (vm == 0 or v == 0):
            vm = 1
            v = 1

        k = (3*close[i]) / vm
        p = p * k

        if p == 0:
            p = 0.00001

        if (close[i] > close[i-1]):
            bp = v
            sp = v / p
        else:
            bp = v / p
            sp = v

        if (abs(bp) > abs(sp)):
            di = sp/bp
        else:
            di = bp/sp

        demand_index.append(di)

    return demand_index


def calculate_dmi(c):
    dmi = []
    std = [np.std(c[i:i+10]) for i in range(len(c))]
    ma = tb.SMA(np.array(std),10)
    std = std[10:]
    ma = ma[10:]
    m = np.median(ma[ma > 0])
    ma[ma == 0] = m
    v = std/ma

    
    for i in range(len(c)):
        value = np.nan
        if i > 10:
            if isnan(v[i-10]):
                period = 5
            else:
                period = int(v[i-10])

            period = min(period, 30)
            period = max(period, 5)
            last_c = c[i-period:i+1]
            value = tb.RSI(last_c,period)[-1]
        dmi.append(value)

    dmi_norm = [dmi[i]/dmi[i-1] if i > 0 and dmi[i-1] > 0 else dmi[i] for i in range(len(dmi))]

    return dmi_norm

def calculate_hma(close,period):
    return tb.WMA(2*tb.WMA(close,period/2) - tb.WMA(close,period), sqrt(period))


def calculate_indicators(df):
    df['ATR'] = tb.ATR(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=15)
    df['CCI'] = tb.CCI(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=23)
    df['CSI'] = calculate_csi(df['High'].values, df['Low'].values, df['Close'].values, df['ATR'], timeperiod=15)
    df['demand_index'] = calculate_demand_index(df['Open'].values,df['High'].values, df['Low'].values, df['Close'].values, df['Volume'], timeperiod=14)
    df['DMI'] = calculate_dmi(df['Close'].values)
    df['EMA'] = tb.EMA(df['Close'].values, timeperiod = 14)
    df['HMA'] = calculate_hma(df['Close'], 20)
    df['MOM'] = tb.MOM(df['Close'], timeperiod=10)
    df['High'] = df['High']/df['Open']
    df['Low'] = df['Low']/ df['Open']
    df['Close'] = df['Close']/ df['Open']

    return df

    
    
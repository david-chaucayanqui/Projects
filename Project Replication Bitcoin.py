#clear all

import quandl
import os
import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from pandas.plotting import autocorrelation_plot
os.chdir('C:/Users/David Chaucayanqui/Documents/Paris 1/M2 Empirical Finance/Semestre2/QMF2/Project')

#Attempting to get data with API but it gives me data after 2014 inclusive. 
#quandl.ApiConfig.api_key = "-------------"
#data = quandl.get("BITSTAMP/USD",start_date="2013-09-03", end_date="2015-02-22")
#data.head(3)
#del data


#####################SECTION 2#############################################

############################trial on the yahoo finance data
#after unsuccessfully attempting to download data from quandl, we attempt to ran the code independently with yahoo finance and coingecko (two different sources)
#working with data from yahoo finance to reinforce the fact that data quality is not available for free, and quandl data may be quite different making all results different

##########################TABLE 1##########################################


bitaudyahoo=pd.read_csv('btc-aud.csv')
bitaudyahoo = bitaudyahoo.rename(columns={'Adj Close': 'priceaud'})

bitrubyahoo=pd.read_csv('btc-rub.csv')
bitrubyahoo = bitrubyahoo.rename(columns={'Adj Close': 'pricerub'})
bitrubyahoo=bitrubyahoo.drop('Date',1)

bitcnyyahoo=pd.read_csv('btc-cny.csv')
bitcnyyahoo = bitcnyyahoo.rename(columns={'Adj Close': 'pricecny'})
bitcnyyahoo=bitcnyyahoo.drop('Date',1)

biteuryahoo=pd.read_csv('btc-eur.csv')
biteuryahoo = biteuryahoo.rename(columns={'Adj Close': 'priceeur'})
biteuryahoo=biteuryahoo.drop('Date',1)

bitgbpyahoo=pd.read_csv('btc-gbp.csv')
bitgbpyahoo = bitgbpyahoo.rename(columns={'Adj Close': 'pricegbp'})
bitgbpyahoo=bitgbpyahoo.drop('Date',1)

bitusdyahoo=pd.read_csv('btc-usd.csv')
bitusdyahoo = bitusdyahoo.rename(columns={'Adj Close': 'priceusd'})
bitusdyahoo=bitusdyahoo.drop('Date',1)

#concatenating all exchange prices into a single dataframe
bitcoinyahoo = pd.concat([bitusdyahoo, bitgbpyahoo,biteuryahoo,bitcnyyahoo,bitrubyahoo,bitaudyahoo], axis=1, sort=False)
bitcoinyahoo = bitcoinyahoo[['priceaud','pricerub','pricecny','priceeur','pricegbp','priceusd','Date']]

##making the date our index
bitcoinyahoo.index = pd.to_datetime(bitcoinyahoo["Date"])
bitcoinyahoo=bitcoinyahoo.drop('Date',1)

del bitusdyahoo, bitgbpyahoo, biteuryahoo, bitcnyyahoo, bitrubyahoo, bitaudyahoo

#calculating return for the bitcoin dataframe
retbityahoo=(bitcoinyahoo/bitcoinyahoo.shift(1)-1)*100
retbityahoo = retbityahoo.drop(retbityahoo.index[[0,1]])


#Table 1 Panel A
#find mean of bitcoin dataframe
retbityahoo.mean()

#find SD of bitcoin dataframe
retbityahoo.std()

#find kurtosis of bitcoin dataframe
retbityahoo.skew()

#find skewness of bitcoin dataframe
retbityahoo.kurtosis()

del bitcoinyahoo, retbityahoo


#download data as csv to coingecko and rename the price column differently for each exchange
bitaud=pd.read_csv('btc-aud-max.csv')
bitaud = bitaud.rename(columns={'price': 'priceaud','snapped_at':'Date'})

bitrub=pd.read_csv('btc-rub-max.csv')
bitrub = bitrub.rename(columns={'price': 'pricerub'})

bitcny=pd.read_csv('btc-cny-max.csv')
bitcny = bitcny.rename(columns={'price': 'pricecny'})

biteur=pd.read_csv('btc-eur-max.csv')
biteur = biteur.rename(columns={'price': 'priceeur'})

bitgbp=pd.read_csv('btc-gbp-max.csv')
bitgbp = bitgbp.rename(columns={'price': 'pricegbp'})

bitusd=pd.read_csv('btc-usd-max.csv')
bitusd = bitusd.rename(columns={'price': 'priceusd'})

#concatenating all exchange prices into a single dataframe
bitcoin = pd.concat([bitaud, bitrub,bitcny,biteur,bitgbp,bitusd], axis=1, sort=False)
bitcoin = bitcoin[['priceaud','pricerub','pricecny','priceeur','pricegbp','priceusd','Date']]

#shifting the price columns since these prices represent the open price for the day, but we want to use the close price. 
#coincidentally, the close price from day 1 is the open price of da 2, so we shift the prices to transform the open price for day 2 to close price day 1
bitcoin.priceaud = bitcoin.priceaud.shift(-1)
bitcoin.pricerub = bitcoin.pricerub.shift(-1)
bitcoin.pricecny = bitcoin.pricecny.shift(-1)
bitcoin.priceeur = bitcoin.priceeur.shift(-1)
bitcoin.pricegbp = bitcoin.pricegbp.shift(-1)
bitcoin.priceusd = bitcoin.priceusd.shift(-1)

#make the date index and drop date column
bitcoin.index = pd.to_datetime(bitcoin["Date"])
bitcoin=bitcoin.drop('Date',1)

del biteur, bitaud, bitcny, bitgbp, bitrub, bitusd

#FIGURE 1
df=bitcoin["priceusd"]
df1 = (df.index >= '2013-07-03') & (df.index <= '2015-02-22')
df2 = df.loc[df1]
plt.plot(df2) 
del df,df1,df2

#FIGURE 2
quandl.ApiConfig.api_key = "QrxMpiQzyrNy-HHrBzWa" 
df5 = quandl.get("BCHAIN/MKTCP", start_date = "2013.01.01", end_date="2015.02.28")

plt.plot(df5) 
plt.title('Total Bitcoin Market Capitalization in USD')
plt.xlabel('Date')
plt.show()

#FIGURE 3
df6 = quandl.get("BCHAIN/NTRAN", start_date = "2013.01.01", end_date="2015.02.28")
plt.plot(df6) 
plt.title('Number of Daily Bitcoin Trades')
plt.xlabel('Date')
plt.show()

del df5, df6

#calculating return for the bitcoin dataframe
retbit=(bitcoin/bitcoin.shift(1)-1)*100
#or same result using: retbit1 = bitcoin.pct_change()

#selecting dates between september 03,2013 and february 22, 2015
retbitc = (retbit.index >= '2013-09-03') & (retbit.index <= '2015-02-22')
returns = retbit.loc[retbitc]

del bitcoin, retbit, retbitc

#TABLE 2
#find mean of bitcoin dataframe
returns.mean()

#find SD of bitcoin dataframe
returns.std()

#find kurtosis of bitcoin dataframe
returns.skew()

#find skewness of bitcoin dataframe
returns.kurtosis()

#Table 1 Panel B
#finding correlation between returns
returns.corr(method='pearson')

#weekly data - leaving only wednesdays
returns["day"] = returns.index.weekday
returnsauto=returns.loc[returns['day'] == 4]


#FIGURE 4
#autocorrelation
returnsauto['priceaud'].autocorr(lag=40)
returnsauto['pricerub'].autocorr(lag=40)
returnsauto['pricecny'].autocorr(lag=40)
returnsauto['pricegbp'].autocorr(lag=40)
returnsauto['priceeur'].autocorr(lag=40)
returnsauto['priceusd'].autocorr(lag=40)
from matplotlib.pyplot import figure
from pandas.plotting import autocorrelation_plot
autocorrelation_plot(returnsauto['priceusd'], label='USD')
autocorrelation_plot(returnsauto['pricerub'], label='RUB')
autocorrelation_plot(returnsauto['pricecny'], label='CNY')
autocorrelation_plot(returnsauto['pricegbp'], label='GBP')
autocorrelation_plot(returnsauto['priceeur'], label='EUR')
autocorrelation_plot(returnsauto['priceaud'], label='AUD')

del returnsauto

##############################SECTION 3################################

######GLS regression and creation of return scaled by volatility#######

#Risk free rate from the FRED website
df3 = pd.read_csv("DGS1MO.csv", index_col=[0], sep=",", parse_dates=True) # 1 Month T bill
df3 = df3.loc["2013-09-03 00:00:00":"2015-02-22",:]
df3 = df3.replace('.', np.nan)
df3=  df3.dropna()
returns["1MoTbill"]= df3["DGS1MO"]
#returns["1MoTbill"] = returns["1MoTbill"].fillna(0.0)
returns=returns.dropna()
returns["1MoTbill"]  =returns["1MoTbill"].astype(float)
returns["excess_rtn"]=returns["priceusd"]-returns["1MoTbill"] 
#During weekend no value of Tbill then we take only days from monday to friday

#returns in weekly fashion
returnsw=returns.loc[returns['day'] == 4]

#rolling SD/ eitehr way works
returns['rolling2_Stdev'] = returns["excess_rtn"].rolling(40).std()    

#second way to calculate rolling SD, needed to make sure i was calculating it with the right answers, it gives the exam same SD
class OnlineVariance(object):
    """
    Welford's algorithm computes the sample variance incrementally.
    """
    def __init__(self, iterable=None, ddof=1):
        self.ddof, self.n, self.mean, self.M2 = ddof, 0, 0.0, 0.0
        if iterable is not None:
            for datum in iterable:
                self.include(datum)

    def include(self, datum):
        self.n += 1
        self.delta = datum - self.mean
        self.mean += self.delta / self.n
        self.M2 += self.delta * (datum - self.mean)
        self.variance = self.M2 / (self.n-self.ddof)

    @property
    def std(self):
        return np.sqrt(self.variance)

ov = OnlineVariance()
for n in range(40):
    ov.include(returns["excess_rtn"].shift(n))

returns['rolling_Stdev'] = ov.std
print(returns)


#creating the excess return scaled by the volatility
returns['retvol']=returns["excess_rtn"]/(returns['rolling2_Stdev'].shift(1))


#creating lagged variables for the return that is scaled by volatility
number_lags = 40
for lag in range(1, number_lags + 1):
    returns['retvollag_' + str(lag)] = returns['retvol'].shift(lag)
    
##################here we dont have enough data to do the sd rolling basis
retvol = returns[returns.columns[11:]]
retvol=retvol.dropna()    #dropping all rows with na because in the regression we dont want the result to be biased by ommited values or having some values being 0 when in reality should no be 0

#FIGURES 5-6
import statsmodels.api as sm
endog=retvol['retvol']
exog=sm.add_constant(retvol[retvol.columns[1:]])
gls_model = sm.GLS(endog, exog)
gls_results = gls_model.fit()
print(gls_results.summary())

#FIGURE 6
#plot t-statistics
tvalue=gls_results.tvalues
tvalue=tvalue.drop(['const'],axis=0)
tvalue=tvalue.to_frame()
tvalue.insert(0, 'Day Lag', range(1, 1 + len(tvalue)))

import matplotlib.pyplot as plt
plt.bar(tvalue['Day Lag'], tvalue[tvalue.columns[1]])
plt.ylabel('t-statistic')
plt.xlabel('Day lag')
plt.title('t-statistics by day')
 
plt.show()



#creating lagged return for equation 3 operations
number_lags = 40
for lag in range(1, number_lags + 1):
    returns['xret_lag' + str(lag)] = returns["excess_rtn"].shift(lag)

retsign = pd.concat([returns['excess_rtn'], returns[returns.columns[52:]]], axis=1, sort=False)
retsign=retsign.dropna()

#assigning -1 and +1
for lag in range(1, number_lags + 1):
    retsign['xret_lag' + str(lag) +'sign'] = np.where((retsign['xret_lag' + str(lag)] >= 0), 1, -1)

#FIGURES 8-9
#GLS with sign +1 or -1
endogsign=retsign['excess_rtn']
exogsign=sm.add_constant(retsign[retsign.columns[41:]])
gls_modelsign = sm.GLS(endogsign, exogsign)
gls_resultssign = gls_modelsign.fit()
print(gls_resultssign.summary())

#FIGURE 10
#plot t-statistics
tvaluesign=gls_resultssign.tvalues
tvaluesign=tvaluesign.drop(['const'],axis=0)
tvaluesign=tvaluesign.to_frame()
tvaluesign.insert(0, 'Day Lag', range(1, 1 + len(tvalue)))

plt.bar(tvaluesign['Day Lag'], tvaluesign[tvaluesign.columns[1]])
plt.ylabel('t-statistic')
plt.xlabel('Day Lag')
plt.title('t-statistics by day')
 
plt.show()

del endog, endogsign, exog, exogsign, lag, n, number_lags, retsign, returnsw, retvol, tvalue, tvaluesign


#######################SECTION 4#############################################

########### added value by Bitcoin to equity only portfolio #################

######### 4.1 Time series momentum trading strategies TSMOM     #############

"""
i=[5,15,25,40,50,75] #5 days=lookback 1week, 15 days=lookback 3 weeks etc...

for x in range (0,5): # Loop to go through the list i
    returns["tsmom_rtn-"+str(i[x])]   = returns["excess_rtn"].rolling(window=i[x]).mean()
    returns["buyposition"+str(i[x])] = np.zeros
    returns["sellposition"+str(i[x])] = np.zeros
    
    for j in range(1,len(returns)) : # Loop to go through the rows of the dataframe "return"
        if returns['tsmom_rtn-'+str(i[x])][j-1] >0:
           returns["buyposition"+str(i[x])][j] = 1
        else:
            returns["buyposition"+str(i[x])][j] = 0
            
        if returns['tsmom_rtn-'+str(i[x])] [j-1]<0:
            returns["sellposition"+str(i[x])][j] = -1
        else:
            returns["sellposition"+str(i[x])][j] = 0
            
"""
         
# Or we can do the same thing without the loops. 
#TABLE 3        
    
#lookback of 1 week (k=1 week= 5 days)
returns["tsmom_rtn-1"]   = returns["excess_rtn"].rolling(window=5).mean()
returns["buyposition1"]  = np.where(returns['tsmom_rtn-1'] >0, 1, 0)   # 1 for buy bitcoin 0 for not entering the market
returns["sellposition1"] = np.where(returns['tsmom_rtn-1'] < 0, -1, 0)# 1 for sell bitcoin 0 for not entering the market
returns["buyposition1"] = returns["buyposition1"].shift(1)
returns["sellposition1"] = returns["sellposition1"].shift(1)
returns["strat_return1"] = returns["buyposition1"] * returns["excess_rtn"] + returns["sellposition1"] * returns["excess_rtn"] #return of the strategy
#Table 2 results
returns["strat_return1"].mean()     #Daily return 
returns["strat_return1"].std()      #Daily sd 
returns["strat_return1"].skew()     #Skewness 
returns["strat_return1"].kurtosis() #Kurtosis 


#lookback of 3 week (k=3weeks= 15days)
returns["tsmom_rtn-3"]   = returns["excess_rtn"].rolling(window=15).mean()
returns["buyposition3"]  = np.where(returns['tsmom_rtn-3'] >0, 1, 0)   # 1 for buy bitcoin 0 for not entering the market
returns["sellposition3"] = np.where(returns['tsmom_rtn-3'] < 0, -1, 0)# 1 for sell bitcoin 0 for not entering the market
returns["buyposition3"] = returns["buyposition3"].shift(1)
returns["sellposition3"] = returns["sellposition3"].shift(1)
returns["strat_return3"] = returns["buyposition3"] * returns["excess_rtn"] + returns["sellposition3"] * returns["excess_rtn"] #return of the strategy
#Table 2 results
returns["strat_return3"].mean()     #Daily return 
returns["strat_return3"].std()      #Daily sd 
returns["strat_return3"].skew()     #Skewness 
returns["strat_return3"].kurtosis() #Kurtosis 


#lookback of 5 week (k=5 weeks= 25 days)
returns["tsmom_rtn-5"]   = returns["excess_rtn"].rolling(window=25).mean()
returns["buyposition5"]  = np.where(returns['tsmom_rtn-5'] >0, 1, 0)   # 1 for buy bitcoin 0 for not entering the market
returns["sellposition5"] = np.where(returns['tsmom_rtn-5'] < 0, -1, 0)# 1 for sell bitcoin 0 for not entering the market
returns["buyposition5"] = returns["buyposition5"].shift(1)
returns["sellposition5"] = returns["sellposition5"].shift(1)
returns["strat_return5"] = returns["buyposition5"] * returns["excess_rtn"] + returns["sellposition5"] * returns["excess_rtn"] #return of the strategy
#Table 2 results
returns["strat_return5"].mean()     #Daily return 
returns["strat_return5"].std()      #Daily sd 
returns["strat_return5"].skew()     #Skewness 
returns["strat_return5"].kurtosis() #Kurtosis 


#lookback of 8 week (k=5 weeks= 40 days)
returns["tsmom_rtn-8"]   = returns["excess_rtn"].rolling(window=40).mean()
returns["buyposition8"]  = np.where(returns['tsmom_rtn-8'] >0, 1, 0)   # 1 for buy bitcoin 0 for not entering the market
returns["sellposition8"] = np.where(returns['tsmom_rtn-8'] < 0, -1, 0)# 1 for sell bitcoin 0 for not entering the market
returns["buyposition8"] = returns["buyposition8"].shift(1)
returns["sellposition8"] = returns["sellposition8"].shift(1)
returns["strat_return8"] = returns["buyposition8"] * returns["excess_rtn"] + returns["sellposition8"] * returns["excess_rtn"] #return of the strategy
#Table 2 results
returns["strat_return8"].mean()     #Daily return 
returns["strat_return8"].std()      #Daily sd 
returns["strat_return8"].skew()     #Skewness 
returns["strat_return8"].kurtosis() #Kurtosis

#lookback of 10 week (k=10 weeks= 50 days)
returns["tsmom_rtn-10"]   = returns["excess_rtn"].rolling(window=50).mean()
returns["buyposition10"]  = np.where(returns['tsmom_rtn-10'] >0, 1, 0)   # 1 for buy bitcoin 0 for not entering the market
returns["sellposition10"] = np.where(returns['tsmom_rtn-10'] < 0, -1, 0)# 1 for sell bitcoin 0 for not entering the market
returns["buyposition10"] = returns["buyposition10"].shift(1)
returns["sellposition10"] = returns["sellposition10"].shift(1)
returns["strat_return10"] = returns["buyposition10"] * returns["excess_rtn"] + returns["sellposition10"] * returns["excess_rtn"] #return of the strategy
#Table 2 results
returns["strat_return10"].mean()     #Daily return 
returns["strat_return10"].std()      #Daily sd 
returns["strat_return10"].skew()     #Skewness 
returns["strat_return10"].kurtosis() #Kurtosis

#lookback of 15 week (k=15 weeks= 75 days)
returns["tsmom_rtn-15"]   = returns["excess_rtn"].rolling(window=75).mean()
returns["buyposition15"]  = np.where(returns['tsmom_rtn-15'] >0, 1, 0)   # 1 for buy bitcoin 0 for not entering the market
returns["sellposition15"] = np.where(returns['tsmom_rtn-15'] < 0, -1, 0)# 1 for sell bitcoin 0 for not entering the market
returns["buyposition15"] = returns["buyposition15"].shift(1)
returns["sellposition15"] = returns["sellposition15"].shift(1)
returns["strat_return15"] = returns["buyposition15"] * returns["excess_rtn"] + returns["sellposition15"] * returns["excess_rtn"] #return of the strategy
#Table 2 results
returns["strat_return15"].mean()     #Daily return 
returns["strat_return15"].std()      #Daily sd 
returns["strat_return15"].skew()     #Skewness 
returns["strat_return15"].kurtosis() #Kurtosis

# Table2 results
ar = np.array([[returns["strat_return1"].mean() , returns["strat_return3"].mean() , returns["strat_return5"].mean() , returns["strat_return8"].mean() , returns["strat_return10"].mean(), returns["strat_return15"].mean()  ], [returns["strat_return1"].std(),returns["strat_return3"].std() ,returns["strat_return5"].std() ,returns["strat_return8"].std(),returns["strat_return10"].std(),returns["strat_return15"].std() ], [returns["strat_return1"].skew(), returns["strat_return3"].skew(), returns["strat_return5"].skew(), returns["strat_return8"].skew(),returns["strat_return10"].skew(),returns["strat_return15"].skew()], [returns["strat_return1"].kurtosis(),returns["strat_return3"].kurtosis(),returns["strat_return5"].kurtosis(),returns["strat_return8"].kurtosis(),returns["strat_return10"].kurtosis(),returns["strat_return15"].kurtosis()]])
Table2 = pd.DataFrame(ar, index = ['Daily return', 'Daily SD', 'Skewness',"Kurtosis"], columns = ['1week', '3weeks', '5weeks', '8weeks',"10weeks","15weeks"])
del ar, Table2



########## 4.2 Diversification enhancement of TSMOM    ####################### 

#TABLE 4
# download S&P500 dataframe from yahoo finance
df1 = pdr.get_data_yahoo("^GSPC", start="2013.08.31", end="2015.02.20")
df1["return"] = df1["Adj Close"].astype(float).pct_change() #sp500 returns
df1["Bitcoin_rtn"]= returns["priceusd"]
df1["TSMOM"]=returns["strat_return15"]
df4=df1.iloc[5:,6:]
df4=df4.dropna()
cols = list(df4)
cols[1], cols[0], cols[2] = cols[0], cols[1], cols[2]
df4= df4.ix[:,cols]
df4.corr(method='pearson')


####################4.3 Mean variance analysis ###############################

##TABLE 5

import pypfopt
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

# We calculate S&P500 returns similarly to TSMOM
df1["1MoTbill"]= df3["DGS1MO"]
df1=df1.dropna()
df1["1MoTbill"]=df1["1MoTbill"].astype(float)
df1["excess_rtn"]=df1["return"]-df1["1MoTbill"] #S&P500 excess returns
df1["spls_rtn-1"]   = df1["excess_rtn"].rolling(window=5).mean()
df1["buyposition1"]  = np.where(df1['spls_rtn-1'] >0, 1, 0)   
df1["sellposition1"] = np.where(df1['spls_rtn-1'] < 0, -1, 0) 
df1["buyposition1"] = df1["buyposition1"].shift(1)
df1["sellposition1"] = df1["sellposition1"].shift(1)
df1["strat_return_sp"] = df1["buyposition1"] * df1["excess_rtn"] + df1["sellposition1"] * df1["excess_rtn"] 

#build the portfolio
df=pd.DataFrame()
df["tsmom"]= returns["strat_return1"]
df["SP"]= df1["strat_return_sp"]
df = df.iloc[5:-1,:]
expected_returns= df.mean()

X = np.stack((df["tsmom"], df["SP"]), axis=0)
X=X.astype(float)
print(np.cov(X))
cov_matrix = np.cov(X)

# Table 5 panel A: Optimise portfolio for efficient risk TSMOM and SP500
ef = EfficientFrontier(expected_returns, cov_matrix, weight_bounds=(0,1), gamma=0) #expected returns mu and risk model S
weights = ef.efficient_risk(target_risk= 0.15, risk_free_rate=returns["1MoTbill"].mean(), market_neutral=False)# maximises Sharpe for a given target risk
ef.portfolio_performance(verbose=True)

df["diff"]=df["tsmom"]-df["tsmom"].mean()
df["diff3"]=df["diff"]**3
num=df["diff3"].sum()
den=310*(df.std()**3)
sk=num/den

df["diff4"]=df["diff"]**4
num1=df["diff4"].sum()
den1=310*(28.80**2)
kr=num1/den1

df2=df

#Table 5 panel B: optimise portfolio for efficient risk SPLS portfolio strategy
df1["buystrat_spls1"] = df1["buyposition1"] * returns["excess_rtn"] 
df1["sellstrat_spls1"] = df1["sellposition1"] * returns["excess_rtn"]

dfsp=pd.DataFrame()
dfsp["buy"]= df1["buystrat_spls1"]
dfsp["sell"]= df1["sellstrat_spls1"]
dfsp=dfsp.iloc[5:,:]
dfsp=dfsp.dropna()
expected_returns_spls =dfsp.mean()

Y = np.stack((dfsp["buy"], dfsp["sell"]), axis=0)
Y=Y.astype(float) 
print(np.cov(Y))
cov_matrix_spls = np.cov(Y)

# Optimise portfolio for efficient risk SP500 Long/Short
ef1 = EfficientFrontier(expected_returns_spls, cov_matrix_spls, weight_bounds=(-1,1), gamma=0) #expected returns mu and risk model S
weights1 = ef1.efficient_risk(target_risk= 0.15, risk_free_rate=returns["1MoTbill"].mean(), market_neutral=False)# maximises Sharpe for a given target risk
ef1.portfolio_performance(verbose=True)

#VaR
import spicy
from scipy.stats import norm

meansptsmom=[5.00,5.60,6.20,6.80,7.40,8.00]
meanls=[0.26,0.36,0.45,0.53,0.62,0.71]
std=[15.00,20,25,30,35,40]
#put negative in front so the values result positive, usually the VaR is in the negative side, but to make it more understandable we make it positive like in the paper
VaR_99ls=-norm.ppf(1-0.99, meanls,std)
VaR_99sptsmom=-norm.ppf(1-0.99, meansptsmom,std)



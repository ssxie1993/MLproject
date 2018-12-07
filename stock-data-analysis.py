# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 22:54:11 2018

@author: 23869
"""


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import datetime as dt

# Any results you write to the current directory are saved as output.
data = pd.read_csv("./all_stocks_5yr.csv")
data.head()

#Display percent of null values
import plotly.graph_objs as go
import plotly.offline as offl

def printNumMissing(allstockdata):
    nummissing = allstockdata.isnull().sum()
    percentmissing = (nummissing.sort_values(ascending=False)/allstockdata.count())*100
    missinganalysis = pd.concat([nummissing, percentmissing], axis=1, keys=["Total items missing", "Percent"])
    print(missinganalysis)

#Plot close data of ticker name
def plotTicker(tickers, allstockdata):
    traces = []

    for ticker in tickers.split(","):
       stockdata = allstockdata[allstockdata.Name == ticker]
       traces += [go.Scatter(x=stockdata.date, y=stockdata.close, name=ticker)]

    layout = dict(title="Closing price vs date" )
    fig = dict(data=traces, layout=layout)
    offl.plot(fig)
    
print ("Analyzing all missing stock data. Dataframe size:", data.shape[0])
printNumMissing(data)


#drops indexes of all stock data that shows null
for colname in data.columns.values:
    data = data.drop((data.loc[data[colname].isnull()]).index)

print ("Checking that all missing data has been removed. Dataframe size:", data.shape[0])
printNumMissing(data)

list_ticker = ['FB', 'AAPL', 'NFLX', 'GOOGL', 'AMZN', 'GM']
#amazon = data[data['Name']=='AMZN'].close
#facebook = data[data['Name']=='FB'].close
#apple = data[data['Name']=='AAPL'].close
#netflix = data[data['Name']=='NFLX'].close
#google = data[data['Name']=='GOOG'].close
#rand_stock = data[data['Name']== list_ticker[len(list_ticker)-1]].close
                  
data.set_index('Name', inplace=True)
amazon = data.loc['AMZN']
amazon.reset_index(inplace=True)
amazon.set_index("date", inplace=True)
amazon = amazon.drop("Name", axis=1)

facebook = data.loc['FB']
facebook.reset_index(inplace=True)
facebook.set_index("date", inplace=True)
facebook = facebook.drop("Name", axis=1)

apple = data.loc['AAPL']
apple.reset_index(inplace=True)
apple.set_index("date", inplace=True)
apple = apple.drop("Name", axis=1)

netflix = data.loc['NFLX']
netflix.reset_index(inplace=True)
netflix.set_index("date", inplace=True)
netflix = netflix.drop("Name", axis=1)

google = data.loc['GOOGL']
google.reset_index(inplace=True)    
google.set_index("date", inplace=True)
google = google.drop("Name", axis=1)

rand_stock = data.loc[list_ticker[len(list_ticker)-1]]
rand_stock.reset_index(inplace=True)    
rand_stock.set_index("date", inplace=True)
rand_stock = rand_stock.drop("Name", axis=1)
                  

#start = datetime(2013, 1, 1)
#end = datetime(2018, 3, 9)

ticker_stocks = pd.concat([facebook, apple,netflix, google, amazon, rand_stock], axis=1,keys=list_ticker)
ticker_stocks.columns.names = ['Ticker','Stock Info']
ticker_stocks.head()
ticker_stocks.reset_index(inplace=True)
ticker_stocks.set_index('date')
ticker_stocks['date'] = pd.to_datetime(ticker_stocks['date'])
ticker_stocks.head()

ticker_stocks.xs(key='close',axis=1,level='Stock Info').max()

returns = pd.DataFrame()
for tick in list_ticker:
    returns[tick+' Return'] = ticker_stocks[tick]['close'].pct_change()
returns.head()
DateCol = ticker_stocks['date']
returns = pd.concat([returns,DateCol], axis = 1)
returns.head()
returns.reset_index(inplace=True)
returns.set_index("date", inplace=True)
returns = returns.drop("index", axis=1)
print(returns)

import seaborn as sns
sns.pairplot(returns[1:])
# dates with the lowest returns for each stock
LowReturnDates = returns.idxmin()
LowReturnDates.head()

returns.idxmax()
returns.std()

import seaborn as sns
whitegrid = sns.set_style('whitegrid')
plt.savefig('whitegrid.pdf')
plt.figure()
heat = sns.heatmap(ticker_stocks.xs(key='close',axis=1,level='Stock Info').corr(),annot=True)
plt.savefig('heatmap.pdf')
plt.figure()

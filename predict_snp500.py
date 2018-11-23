import pandas as pd
import numpy as np
import glob
import functools
import plotly.graph_objs as go
import plotly.offline as offl



#Display percent of null values
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
    

stockinfo = {}

for infile in glob.glob("./stockinfo/*.csv"):
    stockinfo[infile] = pd.read_csv(infile)

#allstockdata = functools.reduce((lambda x,y: pd.concat([x,y])), stockinfo.values())

#Note: run merge.sh before this
allstockdata = pd.read_csv("all_stocks_5yr.csv")

#print (allstockdata)

print ("Analyzing all missing stock data. Dataframe size:", allstockdata.shape[0])
printNumMissing(allstockdata)


#drops indexes of all stock data that shows null
for colname in allstockdata.columns.values:
    allstockdata = allstockdata.drop((allstockdata.loc[allstockdata[colname].isnull()]).index)

print ("Checking that all missing data has been removed. Dataframe size:", allstockdata.shape[0])
printNumMissing(allstockdata)

plotTicker("GOOG,AAL", allstockdata)

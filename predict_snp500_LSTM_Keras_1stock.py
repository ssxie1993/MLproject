# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 23:16:59 2018

@author: 23869
"""

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers import LSTM,Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "./input"]).decode("utf8"))

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

ticker  = 'AMZN'
plotTicker(ticker, data)
#plotTicker("GOOGL", data)

stock1 = data[data['Name']== ticker].close
#cl_stock2 = data[data['Name']=='GOOGL'].close
scl = MinMaxScaler()
#Scale the data
stock1= np.array(stock1)
stock1 = stock1.reshape(stock1.shape[0],1)
stock1 = scl.fit_transform(stock1)

#cl_stock2= np.array(cl_stock2)
#cl_stock2 = cl_stock2.reshape(cl_stock2.shape[0],1)
#cl_stock2 = scl.fit_transform(cl_stock2)

#Create a function to process the data into "step" day look back slices
def processData(data,lb):
    X,Y = [],[]
    for i in range(len(data)-lb-1):
        X.append(data[i:(i+lb),0])
        Y.append(data[(i+lb),0])
    return np.array(X),np.array(Y)
step = 60
X,y = processData(stock1,step)
#X_stock2,y_stock2 = processData(cl_stock2,step)
split = 0.8
#X_train,X_test = X[:int(X.shape[0]*0.80)],X[int(X.shape[0]*0.80):]
X_train = X[:int(X.shape[0]*split)]
y_train = y[:int(y.shape[0]*split)]
X_test = X[int(X.shape[0]*split):]
y_test = y[int(y.shape[0]*split):]
#print(X_train.shape[0])
#print(X_test.shape[0])
#print(y_train.shape[0])
#print(y_test.shape[0])

#X_test = X_stock2[:int(X_stock2.shape[0]*split)]
#y_test = y_stock2[:int(y_stock2.shape[0]*split)]

print(X_train.shape[0])
print(X_test.shape[0])
print(y_train.shape[0])
print(y_test.shape[0])


#Build the model
model = Sequential()
#model.add(LSTM(256,input_shape=(1,step)))
model.add(LSTM(256,input_shape=(step,1)))
model.add(Dense(1))
model.compile(optimizer='adam',loss='mse')
#Reshape data for (Sample,Timestep,Features) 
X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],1))
X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],1))
#Fit model with history to check for overfitting
history = model.fit(X_train,y_train,epochs=150,validation_data=(X_test,y_test),shuffle=False)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
#plt.title('AMZN: Train 80%, Test 20%')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['train','validation'])
plt.show() 
#We see this is pretty jumpy but we will keep it at 300 epochs. With more data, it should smooth out the loss
#Lets look at the fit
Xt = model.predict(X_test)
plt.plot(scl.inverse_transform(y_test.reshape(-1,1)))
plt.plot(scl.inverse_transform(Xt))
plt.show() 

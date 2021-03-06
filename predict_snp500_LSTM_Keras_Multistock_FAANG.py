

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

#Create a function to process the data into "step" day look back slices
def processData(data,lb):
    X,Y = [],[]
    for i in range(len(data)-lb-1):
        X.append(data[i:(i+lb),0])
        Y.append(data[(i+lb),0])
    return np.array(X),np.array(Y)

step = 60
n_epochs = 10;
#Build the model
model = Sequential()
#model.add(LSTM(256,input_shape=(1,step)))
model.add(LSTM(256,input_shape=(step,1)))
model.add(Dense(1))
model.compile(optimizer='adam',loss='mse')

#plotTicker("AAPL", data)
#plotTicker("GOOGL", data)
list_ticker = ['FB', 'AAPL', 'NFLX', 'GOOG', 'AMZN']
test_stock = data[data['Name']=='AMZN'].close
stock2 = data[data['Name']=='FB'].close
stock3 = data[data['Name']=='AAPL'].close
stock4 = data[data['Name']=='NFLX'].close
stock5 = data[data['Name']=='GOOG'].close

for i in range(len(list_ticker)-1):
    ticker = list_ticker[i]
    print ('Train using ', ticker)
    stock1 = data[data['Name']==ticker].close             
    scl = MinMaxScaler()
    #Scale the data
    stock1= np.array(stock1)
    stock1 = stock1.reshape(stock1.shape[0],1)
    stock1 = scl.fit_transform(stock1)
    
    test_stock= np.array(test_stock)
    test_stock = test_stock.reshape(test_stock.shape[0],1)
    test_stock = scl.fit_transform(test_stock)
    
    X,y = processData(stock1,step)
    X_test_stock,y_test_stock = processData(test_stock,step)
    train_split = 1.00
    #X_train,X_test = X[:int(X.shape[0]*0.80)],X[int(X.shape[0]*0.80):]
    X_train = X[:int(X.shape[0]*train_split)]
    y_train = y[:int(y.shape[0]*train_split)]
    #X_test = X[int(X.shape[0]*split):]
    #y_test = y[int(y.shape[0]*split):]
    #print(X_train.shape[0])
    #print(X_test.shape[0])
    #print(y_train.shape[0])
    #print(y_test.shape[0])
    test_split = 0.80
    X_test_stock = X_test_stock[int(X_test_stock.shape[0]*test_split):]
    y_test_stock = y_test_stock[int(y_test_stock.shape[0]*test_split):]

#    X_test_stock = X_test_stock[:int(X_test_stock.shape[0]*test_split)]
#    y_test_stock = y_test_stock[:int(y_test_stock.shape[0]*test_split)]

#    print(X_train.shape[0])
#    print(y_train.shape[0])


    X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],1))   
    X_test_stock = X_test_stock.reshape((X_test_stock.shape[0],X_test_stock.shape[1],1))
    history = model.fit(X_train,y_train,epochs=n_epochs,validation_data=(X_test_stock,y_test_stock),shuffle=False)
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(['train','validation'])
    plt.show() 
    #We see this is pretty jumpy but we will keep it at 300 epochs. With more data, it should smooth out the loss
    #Lets look at the fit
    Xt = model.predict(X_test_stock)
    plt.plot(scl.inverse_transform(y_test_stock.reshape(-1,1)))
    plt.plot(scl.inverse_transform(Xt))
    plt.show() 

#############################################################################
#############################################################################
#############################################################################
#Now Last train with 80% of AMZN
ticker = list_ticker[len(list_ticker)-1]
print ('Train using ', ticker)
stock1 = data[data['Name']==ticker].close      
stock1= np.array(stock1)
stock1 = stock1.reshape(stock1.shape[0],1)
stock1 = scl.fit_transform(stock1)

test_stock= np.array(test_stock)
test_stock = test_stock.reshape(test_stock.shape[0],1)
test_stock = scl.fit_transform(test_stock)

X,y = processData(stock1,step)
X_test_stock,y_test_stock = processData(test_stock,step)
train_split = 0.80
#X_train,X_test = X[:int(X.shape[0]*0.80)],X[int(X.shape[0]*0.80):]
X_train = X[:int(X.shape[0]*train_split)]
y_train = y[:int(y.shape[0]*train_split)]
#X_test = X[int(X.shape[0]*split):]
#y_test = y[int(y.shape[0]*split):]
#print(X_train.shape[0])
#print(X_test.shape[0])
#print(y_train.shape[0])
#print(y_test.shape[0])
test_split = train_split
X_test_stock = X_test_stock[int(X_test_stock.shape[0]*test_split):]
y_test_stock = y_test_stock[int(y_test_stock.shape[0]*test_split):]

X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],1))   
X_test_stock = X_test_stock.reshape((X_test_stock.shape[0],X_test_stock.shape[1],1))
history = model.fit(X_train,y_train,epochs=n_epochs,validation_data=(X_test_stock,y_test_stock),shuffle=False)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['train','validation'])
plt.show() 
#We see this is pretty jumpy but we will keep it at 300 epochs. With more data, it should smooth out the loss
#Lets look at the fit
Xt = model.predict(X_test_stock)
plt.plot(scl.inverse_transform(y_test_stock.reshape(-1,1)))
plt.plot(scl.inverse_transform(Xt))
plt.show() 
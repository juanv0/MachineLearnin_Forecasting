import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
#python3 -m pip


df = pd.read_csv("rawdata.txt")


class toTimeSeries():

	def __init__(self, df):
		self.df = df
		
	def toTimeS(self, n_in=1):
		cols, names = list(), list()
		
		for i in range(1, n_in+1):
			cols.append(self.df["Births"].shift(-i)) 
			names.append("Input value {}".format(i))

		cols.append(self.df["Births"].shift(-(n_in+1)))
		names.append("1_Lag_Day_Births")
		df.drop(["Births"], axis= 1)
		self.df = pd.concat(cols,axis=1)
		self.df.columns=names
		return (self.df.dropna())


def pred(df, n_in, test_l = 6):
	train_length = test_l
	y = df["1_Lag_Day_Births"].values
	X = df.drop(["1_Lag_Day_Births"], axis =1).values
	x_train, x_test = X[:len(X)- train_length], X[len(X)- train_length:]
	y_train, y_test = y[:len(y)- train_length], y[len(y)- train_length:]
	model = Sequential()
	model.add(Dense(12, activation='relu', input_dim=n_in))
	model.add(Dense(8, activation='relu'))
	model.add(Dense(1))
	model.compile(optimizer='adam', loss='mse')
	model.fit(x_train, y_train, epochs=1000, batch_size=2, verbose=2)
	trainScore = model.evaluate(x_train, y_train, verbose=0)
	testScore = model.evaluate(x_test, y_test, verbose=0)
	trainPredict = model.predict(x_train)
	testPredict = model.predict(x_test)
	return trainPredict, testPredict
	
	
obj = toTimeSeries(df)
n_in = 60
test_l = 120
dataset=df.values
trainPredict, testPredict = [],[]
trainPredict, testPredict = pred(obj.toTimeS(n_in), n_in, test_l=test_l)
y = df["Date"].values
yTrainPredictPlot = y[n_in:len(y)-(test_l)-1]
yTestPredictPlot = y[len(y)-(test_l):len(y)]
plt.plot(df["Date"],df["Births"])
plt.plot(yTestPredictPlot, testPredict)
plt.plot(yTrainPredictPlot, trainPredict)
plt.show()
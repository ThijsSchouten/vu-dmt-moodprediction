# LSTM model for mood prediction using the window method (with memory)
# according to: https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/

import math
import matplotlib.pyplot as plt
import numpy
from data import *
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset.iloc[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset.iloc[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)

data = preprocess_raw_data()
mood = data.loc[:, (slice(None), 'mood')] # mood waardes van alle patienten
train_size = int(len(mood) * 0.67)
test_size = len(mood) - train_size
# mood is nu geordend per patient, dus de test set is nu alleen laatste paar patienten = niet goed
train, test = mood.iloc[0:train_size,:], mood.iloc[train_size:len(mood),:]
# reshape into X=t and Y=t+1
look_back = 3
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
########################################################################################################################
# Moet nog gebeuren
########################################################################################################################
# # reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
print(trainX.shape)
# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
# Die scaler doet nu niks, want is al genormalized
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(mood)
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
#calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(mood)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(mood)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(mood)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(mood))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
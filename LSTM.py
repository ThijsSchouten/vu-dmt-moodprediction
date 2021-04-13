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
	for i in range(len(dataset) - look_back - 1):
		a = dataset.iloc[i:(i + look_back), 0]
		dataX.append(a)
		dataY.append(dataset.iloc[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)


def create_traintest(mood, look_back = 3):
	train_size = int(len(mood) * 0.67)
	test_size = len(mood) - train_size
	# mood is nu geordend per patient, dus de test set is nu alleen laatste paar patienten = niet goed
	train, test = mood.iloc[0:train_size, :], mood.iloc[train_size:len(mood), :]
	# reshape into X=t and Y=t+1
	trainX, trainY = create_dataset(train, look_back)
	testX, testY = create_dataset(test, look_back)
	# reshape input to be [samples, time steps, features]
	trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
	testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
	return trainX, trainY, testX, testY


def create_LSTM(trainX,trainY,look_back):
	# create and fit the LSTM network
	model = Sequential()
	model.add(LSTM(4, input_shape=(1, look_back)))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
	return model

def invert_predictions(scaler, mood,trainPredict,trainY,testPredict,testY):
	# invert predictions
	# Die  minmaxscaler en fit doen nu niks, want is al genormalized, maar is wel nodig voor praktijk
	dataset = scaler.fit_transform(mood)
	trainPredict = scaler.inverse_transform(trainPredict)
	trainY = scaler.inverse_transform([trainY])
	testPredict = scaler.inverse_transform(testPredict)
	testY = scaler.inverse_transform([testY])
	return trainPredict, trainY, testPredict, testY

def shift_predictions_for_plots(mood, trainPredict, look_back, testPredict):
	# shift train predictions for plotting
	trainPredictPlot = numpy.empty_like(mood)
	trainPredictPlot[:, :] = numpy.nan
	trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict
	# shift test predictions for plotting
	testPredictPlot = numpy.empty_like(mood)
	testPredictPlot[:, :] = numpy.nan
	testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(mood) - 1, :] = testPredict
	return trainPredictPlot, testPredictPlot


def plot_baseline_vs_predictions(scaler, mood,trainPredictPlot,testPredictPlot):
		# plot baseline and predictions
		plt.plot(scaler.inverse_transform(mood))
		plt.plot(trainPredictPlot)
		plt.plot(testPredictPlot)
		plt.show()


data = preprocess_raw_data()
mood = data.loc[:, (slice(None), 'mood')]  # mood waardes van alle patienten
look_back = 3							   # aantal dagen dat je terugkijkt
trainX, trainY, testX, testY = create_traintest(mood)
model = create_LSTM(trainX,trainY,look_back)
# Make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
scaler = MinMaxScaler(feature_range=(0, 1))
# Invert predictions
trainPredict, trainY, testPredict, testY = invert_predictions(scaler, mood,trainPredict,trainY,testPredict,testY)
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
print('Test Score: %.2f RMSE' % (testScore))
trainPredictPlot, testPredictPlot = shift_predictions_for_plots(mood, trainPredict, look_back, testPredict)
plot_baseline_vs_predictions(scaler, mood,trainPredictPlot,testPredictPlot)
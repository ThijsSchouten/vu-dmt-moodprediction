# LSTM model for mood prediction using the window method (with memory)
# according to: https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/

import math
import matplotlib.pyplot as plt
import numpy
import statistics
from data import *
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor


def create_dataset(dataset, look_back=3):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset.iloc[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset.iloc[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)


def create_traintest(mood, look_back=3):
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


def create_LSTM(trainX, trainY, look_back, epochs=10, batch_size=1):
    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=2)
    return model


def inverse_normalization2(trainPredict, trainY, testPredict, testY, scaler_fp='scalers/scaler.pkl'):
    """
	Loads scaler and applies inverserve
	normalisation to labels.
	"""
    scaler = load(open(scaler_fp, 'rb'))
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


def plot_baseline_vs_predictions(mood, trainPredictPlot, testPredictPlot, scaler_fp='scalers/scaler.pkl'):
    # plot baseline and predictions
    scaler = load(open(scaler_fp, 'rb'))
    plt.plot(scaler.inverse_transform(mood))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()


def create_traintest(mood, look_back=3):
    train_size = int(len(mood) * 1)
    # mood is nu geordend per patient, dus de test set is nu alleen laatste paar patienten = niet goed
    train = mood.iloc[0:train_size, :]
    # reshape into X=t and Y=t+1
    trainX, trainY = create_dataset(train, look_back)
    # reshape input to be [samples, time steps, features]
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    return trainX, trainY


data = preprocess_raw_data()
ids = data.index.levels[0].values
mood = data.loc[:, (slice(None), 'mood')]  # mood waardes van alle patienten
look_back = 1  # aantal dagen dat je terugkijkt
tscv = TimeSeriesSplit()
train_results = dict()
test_results = dict()
for id in ids:
    trainX, trainY = create_traintest(mood.loc[id], look_back)
    batch_size = [1, 2, 3]
    epochs = [5, 10, 15]
    model = KerasRegressor(build_fn = create_LSTM(trainX, trainY, look_back), verbose =1)
    param_grid = dict(epochs=epochs, batch_size=batch_size)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=tscv, verbose=0)
    resultgridsearch = grid.fit(trainX, trainY)
    best_params = clf.best_params_
    model = KerasRegressor(create_LSTM(trainX, trainY, look_back),best_params)
# # Make predictions
# trainPredict = model.predict(trainX)
# testPredict = model.predict(testX)
# # Invert predictions
# trainPredict, trainY, testPredict, testY = inverse_normalization2(trainPredict,trainY,testPredict,testY)
# # calculate root mean squared error
# trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
# #print('Train Score: %.2f RMSE' % (trainScore))
# testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
# #print('Test Score: %.2f RMSE' % (testScore))
# train_results[id] = [trainScore, train_index]
# test_results[id] = [testScore, test_index]
# 	# trainPredictPlot, testPredictPlot = shift_predictions_for_plots(mood, trainPredict, look_back, testPredict)
# 	# plot_baseline_vs_predictions(mood,trainPredictPlot,testPredictPlot)

#
# print(statistics.mean(train_results.values()), statistics.mean(test_results.values()))
# print(statistics.variance(train_results.values()), statistics.variance(test_results.values()))


########################################################################################################################
# Graveyard
# for train_index, test_index in tscv.split(mood.loc[id]):
# 	days = numpy.array(list(range(len(mood.loc[id]) + 1))[1:])
# 	print("TRAIN:", train_index, "TEST:", test_index)
# 	trainX, testX = mood.loc[id].iloc[train_index], mood.loc[id].iloc[test_index]
# 	trainY, testY = days[train_index], days[test_index]
# create model per id per cv loop

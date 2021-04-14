# LSTM model for mood prediction using the window method (with memory)
# according to: https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/

import math
import matplotlib.pyplot as plt
import numpy
import statistics
from data import *
from svm import *
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor


LOOK_BACK = 3 # Constant for how many days there should be looked back

def create_dataset(dataset, look_back= LOOK_BACK):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset.iloc[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset.iloc[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)


def create_traintest(mood, look_back= LOOK_BACK):
    train_size = int(len(mood) * 0.75)
    # mood is nu geordend per patient, dus de test set is nu alleen laatste paar patienten = niet goed
    train, test = mood.iloc[0:train_size, :], mood.iloc[train_size:len(mood), :]
    # reshape into X=t and Y=t+1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    # reshape input to be [samples, time steps, features]
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    return trainX, trainY, testX, testY


def create_LSTM(look_back = LOOK_BACK):
    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def inverse_normalization2(trainPredict, trainY, testPredict, testY, scaler_fp='scalers/scaler.pkl'):
    """
	Loads scaler and applies inverserve
	normalisation to labels.
	"""
    scaler = load(open(scaler_fp, 'rb'))
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform(trainY)
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform(testY)
    return trainPredict, trainY, testPredict, testY


def shift_predictions_for_plots(mood, trainPredict, testPredict, look_back = LOOK_BACK):
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

data = preprocess_raw_data()
ids = data.index.levels[0].values
mood = data.loc[:, (slice(None), 'mood')]  # mood waardes van alle patienten
tscv = TimeSeriesSplit()
train_results = dict()
test_results = dict()
for id in ids:
    trainX, trainY, testX, testY = create_traintest(mood.loc[id])
    batch_size = [2, 4, 8]
    epochs = [10, 20, 40]
    param_grid = dict(epochs=epochs, batch_size=batch_size)
    model = KerasRegressor(build_fn = create_LSTM, verbose =0)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=tscv, verbose=0)
    resultgridsearch = grid.fit(trainX, trainY)
    best_params = resultgridsearch.best_params_
    print(best_params)
    # Make predictions
    trainPredict = resultgridsearch.predict(trainX)
    testPredict = resultgridsearch.predict(testX)
    # Reshape data
    trainY = trainY.reshape(-1, 1)
    testY = testY.reshape(-1, 1)
    trainPredict = trainPredict.reshape(-1,1)
    testPredict = testPredict.reshape(-1,1)
    # Invert predictions
    trainPredict, trainY, testPredict, testY = inverse_normalization2(trainPredict, trainY, testPredict, testY)
    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
    #print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY, testPredict))
    #print('Test Score: %.2f RMSE' % (testScore))
    train_results[id] = trainScore
    test_results[id] = testScore
    # # trainPredictPlot, testPredictPlot = shift_predictions_for_plots(mood, trainPredict, testPredict)
    # # plot_baseline_vs_predictions(mood,trainPredictPlot,testPredictPlot)

print('Train Score: %.2f RMSE' % (statistics.mean(train_results.values())), 'Test Score: %.2f RMSE' % (statistics.mean(test_results.values())))
print('Train Variance: %.2f RMSE' % (statistics.variance(train_results.values())), 'Test Variance: %.2f RMSE' % (statistics.variance(test_results.values())))


########################################################################################################################
# Graveyard
# for train_index, test_index in tscv.split(mood.loc[id]):
# 	days = numpy.array(list(range(len(mood.loc[id]) + 1))[1:])
# 	print("TRAIN:", train_index, "TEST:", test_index)
# 	trainX, testX = mood.loc[id].iloc[train_index], mood.loc[id].iloc[test_index]
# 	trainY, testY = days[train_index], days[test_index]
# create model per id per cv loop

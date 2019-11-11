import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,  LSTM
from keras import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv('nikkei.csv', encoding="shift_jis")
# df.head()
Hi = np.array([df.iloc[:, 2]])
Low = np.array([df.iloc[:, 3]])
Close = np.array([df.iloc[:, 4]])

plt.figure(1)
H, = plt.plot(Hi[0, :])
L, = plt.plot(Low[0, :])
C, = plt.plot(Close[0, :])

plt.legend([H, L, C], ["High", "Low", "CLose"])
plt.show(block=False)

X = np.concatenate([Hi, Low], axis=0)
X = np.transpose(X)

Y = Close
Y = Y.reshape(-1, 1)

scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)

scaler1 = MinMaxScaler()
scaler1.fit(Y)
Y = scaler1.transform(Y)

X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
print(X.shape)

X_train = X[:190, :, :]
X_test = X[190:, :, :]
Y_train = Y[:190, :]
Y_test = Y[190:, :]
model = Sequential()
model.add(LSTM(100, activation='tanh', input_shape=(1, 2), recurrent_activation='hard_sigmoid'))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=[metrics.mae])
model.fit(X_train, Y_train, epochs=50, batch_size=1, verbose=2)

Predict = model.predict(X_test, verbose=1)
# print(Predict)
plt.figure(figsize=(15, 10))
plt.plot(Y_test, label='Test')
plt.plot(Predict, label='Prediction')
plt.legend(loc='best')
plt.show()


Y_train = scaler1.inverse_transform(Y_train)
Y_train = pd.DataFrame(Y_train)
Y_train.index = pd.to_datetime(df.iloc[:190,0])

Y_test = scaler1.inverse_transform(Y_test)
Y_test = pd.DataFrame(Y_test)
Y_test.index = pd.to_datetime(df.iloc[190:,0])

# Predict = model.predict(X_test)
Predict = scaler1.inverse_transform(Predict)
print(Predict)
Predict = pd.DataFrame(Predict)
Predict.index=pd.to_datetime(df.iloc[190:,0])

plt.figure(figsize=(15,10))
plt.plot(Y_test,label = 'Test')
plt.plot(Predict, label = 'Prediction')
plt.legend(loc='best')
plt.show()
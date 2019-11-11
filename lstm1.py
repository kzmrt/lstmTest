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
L = len(df)
Y = df.iloc[:, 4]  # 終値の列のみ抽出する。
Y = np.array(Y)  # numpy配列に変換する。
# print(Y)
Y = Y.reshape(-1, 1)  # 行列に変換する。（配列の要素数行×1列）
# print(Y)
# plt.plot(Y)
# plt.show(block= False)

X1 = Y[0:L-3, :]  # 予測対象日の3日前のデータ
# print(X1)
X2 = Y[1:L-2, :]  # 予測対象日の2日前のデータ
# print(X2)
X3 = Y[2:L-1, :]  # 予測対象日の前日データ
# print(X3)
Y = Y[3:L, :]  # 予測対象日のデータ
# print(Y)
X= np.concatenate([X1, X2, X3], axis=1)  # numpy配列を結合する。
# print(X)
print(f'X shape is {X.shape}')
print(f'Y shape is {Y.shape}')

scaler = MinMaxScaler()  # データを0～1の範囲にスケールするための関数。
scaler.fit(X)  # スケーリングに使用する最小／最大値を計算する。
X = scaler.transform(X)  # Xをを0～1の範囲にスケーリングする。
# print(X)

scaler1 = MinMaxScaler()  # データを0～1の範囲にスケールするための関数。
scaler1.fit(Y)  # スケーリングに使用する最小／最大値を計算する。
Y = scaler1.transform(Y)  # Yをを0～1の範囲にスケーリングする。
# print(Y)

X = np.reshape(X, (X.shape[0], 1, X.shape[1]))  # 3次元配列に変換する。
# print(X)

# train, testデータを定義
X_train = X[:190, :, :]
X_test = X[190:, :, :]
Y_train = Y[:190, :]
Y_test = Y[190:, :]

model = Sequential()
model.add(LSTM(10, activation = 'tanh', input_shape = (1,3), recurrent_activation= 'hard_sigmoid'))
model.add(Dense(1))

model.compile(loss= 'mean_squared_error', optimizer = 'rmsprop', metrics=[metrics.mae])
model.fit(X_train, Y_train, epochs=100, verbose=2)
Predict = model.predict(X_test)

# plt.figure(figsize=(15,10))
# plt.plot(Y_test, label = 'Test')
# plt.plot(Predict, label = 'Prediction')
# plt.legend(loc='best')
# plt.show()

# オリジナルのスケールに戻す、タイムインデックスを付ける。
Y_train = scaler1.inverse_transform(Y_train)
Y_train = pd.DataFrame(Y_train)
Y_train.index = pd.to_datetime(df.iloc[3:193,0])

Y_test = scaler1.inverse_transform(Y_test)
Y_test = pd.DataFrame(Y_test)
Y_test.index = pd.to_datetime(df.iloc[193:,0])

# Predict = model.predict(X_test)
Predict = scaler1.inverse_transform(Predict)
Predict = pd.DataFrame(Predict)
Predict.index=pd.to_datetime(df.iloc[193:,0])

plt.figure(figsize=(15,10))
plt.plot(Y_test, label = 'Test')
plt.plot(Predict, label = 'Prediction')
plt.legend(loc='best')
plt.show()
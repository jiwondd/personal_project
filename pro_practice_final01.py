import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
from tensorflow.keras.layers import Bidirectional
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import MaxAbsScaler,RobustScaler
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
import time
from sklearn.model_selection import train_test_split
import inspect, os

path='./_beach/'
weather = pd.read_csv(path + '해운대 날씨.csv')
customer = pd.read_csv(path + '해운대 입장객수2.csv')

# 날짜를 date type으로 변경 후, 나머지는 numeric type으로 변경
weather['날짜'] = pd.to_datetime(weather['날짜'], infer_datetime_format=True)
weather.iloc[:,1:] = weather.iloc[:,1:].apply(pd.to_numeric)

customer['방문일'] = pd.to_datetime(customer['방문일'], infer_datetime_format=True)
customer['방문객수'] = customer['방문객수'].str.replace(",","")
customer.iloc[:,1:] = customer.iloc[:,1:].apply(pd.to_numeric)

# merge data : 방문객수 예측을 위한 데이터
total_data = pd.merge(weather, customer, left_on='날짜', right_on="방문일", how='inner')
total_data = total_data[['강수_관측값', "기온", "습도", "체감온도", "평균풍속", "평균기압", "평균수온", "평균최대파고", "평균파주기", "방문객수"]]

# 결측치 처리하기
total_data.loc[total_data['평균수온'] != total_data['평균수온'], '평균수온'] = total_data['평균수온'].mean()
total_data.loc[total_data['평균풍속'] != total_data['평균풍속'], '평균풍속'] = total_data['평균풍속'].mean()
total_data.loc[total_data['평균기압'] != total_data['평균기압'], '평균기압'] = total_data['평균기압'].mean()
total_data.loc[total_data['평균최대파고'] != total_data['평균최대파고'], '평균최대파고'] = total_data['평균최대파고'].mean()
total_data.loc[total_data['평균파주기'] != total_data['평균파주기'], '평균파주기'] = total_data['평균파주기'].mean()
total_data = total_data.fillna(0)
print(total_data.isnull().sum())

input_set = total_data[["강수_관측값", "기온", "습도", "체감온도", "평균풍속", "평균기압", "평균수온", "평균최대파고", "평균파주기", "방문객수"]]
output_set = total_data[['강수_관측값', '기온', "방문객수"]]

# print(input_set.shape,output_set.shape) #(1465, 10) (1465, 3)


# train_x, test_x, train_y, test_y = train_test_split(total_data.iloc[:, :-1], total_data['방문객수'], test_size=0.2)
train_x, test_x, train_y, test_y = train_test_split(input_set, output_set, test_size=0.2)

print(train_x.shape,test_x.shape) #(1172, 10) (293, 10)
print(test_x.info)

scaler = MinMaxScaler()
train_x_scaled = scaler.fit_transform(train_x)
test_x_scaled = scaler.transform(test_x)

train_x=train_x_scaled.reshape(1172, 5, 2)
test_x=test_x_scaled.reshape(293, 5, 2)

train_y = train_y.astype(float)
test_y = test_y.astype(float)

print(train_x.shape, test_x.shape) #(1172, 5, 2) (293, 5, 2)

# 2. 모델구성
model=Sequential()
model.add(LSTM(units=256,input_shape=(5,2),return_sequences=True))
model.add(Bidirectional(LSTM(512)))
model.add(Dense(256, activation='swish'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='swish'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='swish'))
model.add(Dense(3))
model.summary()

# 3. 컴파일 훈련
earlyStopping=EarlyStopping(monitor='val_loss',patience=500,mode='min',verbose=1,restore_best_weights=True) 
model.compile(loss='mse',optimizer='adam')
hist=model.fit(train_x,train_y,epochs=3000,batch_size=1024,validation_split=0.2, callbacks=[earlyStopping])


# 4. 평가, 예측
loss=model.evaluate(test_x,test_y)
pred_y=model.predict(test_x)

print(pred_y)

print('loss: ',loss)
print('예상 강수량, 예상 기온, 예상 방문객 수: ', pred_y[-1:])

'''
loss:  3390305.75 <- swish 섞음
loss:  4404846592.0 <-노드수 확 늘려봄
loss:  3270562560.0<-다시 노드 줄임
loss:  5216742.0<-노드 더 줄임
loss:  4358980.5<-swish 더 넣음
loss:  413259.53125 <-patience 100->300 늘림
loss:  223695.875 <-patience 300->500 늘림
loss:  193260.5 <-vla_split 0.1->0.2


'''
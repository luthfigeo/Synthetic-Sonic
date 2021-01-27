import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D,  AveragePooling1D, LeakyReLU

from tensorflow.keras.metrics import RootMeanSquaredError
import tensorflow as tf

db = pd.read_csv('/data.csv')
db['NPHI'] = np.where(db['NPHI']<0, np.nan, db['NPHI'])
#db['NPHI'] = np.where(db['NPHI']>0.5, np.nan, db['NPHI'])
db['GR'] = np.where(db['GR']<0, np.nan, db['GR'])
#db['GR'] = np.where(db['GR']>300, np.nan, db['GR'])
db['DRES'] = np.where(db['DRES']<0, np.nan, db['DRES'])
db['DRES'] = np.where(db['DRES']>1000, np.nan, db['DRES'])
db['logDRES'] = np.log(db['DRES'])
well = db['WELL'].unique()
print(len(well))
db = db[['DEPT','WELL','GR','NPHI','PEF','RHOB','DRES','SP','CAL','DTCO','DTSM']].dropna()

X = db.iloc[:,2:-1]
Y = db.iloc[:,-1]

test = 0.20
seed = 0
batch = 500
X_train, X_val, Y_train, Y_val = model_selection.train_test_split(X, Y, test_size=test, random_state=seed)

def ModelNN():
  classifier = Sequential([
                           #Dense(25, activation = 'relu', input_dim=5),
                           #BatchNormalization(),
                           LSTM(64, input_shape=(8,1), activation='tanh', return_sequences=True),
                           Conv1D(32,8,activation='tanh', strides=2, padding='same'),
                           AveragePooling1D(pool_size=3,strides=1, padding='same'),
                           Dense(12,activation = 'relu'),
                           LeakyReLU(alpha=0.01),
                           Dense(1),
                           #Dropout(0.2)                     
        ])
  classifier.compile(optimizer='adam', loss='huber', metrics=[RootMeanSquaredError(), 'mean_squared_error'])
  return classifier

X_train = np.expand_dims(X_train, 2)
X_val = np.expand_dims(X_val, 2)
X = np.expand_dims(X, 2) 
classifier = ModelNN()

def fitmodel():
  with tf.device('/device:GPU:0'):
    history = classifier.fit(X_train,Y_train, epochs=10, batch_size=50, verbose=2)
    Y_predict_test = classifier.predict(X_val)
    Y_predict_train = classifier.predict(X_train)
    return history, Y_predict_test, Y_predict_train

history, Y_predict_test, Y_predict_train = fitmodel()

mse_train = mean_squared_error(Y_predict_train.reshape(-1,Y_predict_train.shape[1]).min(axis=1),Y_train, squared=False)
mse_val = mean_squared_error(Y_predict_test.reshape(-1,Y_predict_test.shape[1]).min(axis=1),Y_val, squared=False)
r2_val = r2_score(Y_predict_test.reshape(-1,Y_predict_test.shape[1]).min(axis=1),Y_val)
r2_all = r2_score(db['DTSM_pred'],db['DTSM'])
print('rmse_train= ',mse_train,'\nrmse_test= ',mse_val,'\nr2 val= ',r2_val,'\nr2 all= ',r2_all)
# Stock-price-prediction1
!pip install yfinance
!pip install tensorflow
!pip install sklearn

'''import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import datetime

code = input("Enter the stock code: ")

ticker = code  # You can change this to any stock symbol
start_date = input("Enter a start date in YYYY-MM-DD format: ")
end_date  = input("Enter a end date in YYYY-MM-DD format: ")
if (start_date >= end_date):
  print("Invalid Time")
  exit()
else:
  print("Valid Time")
 
  df = yf.download(ticker, start=start_date, end=end_date)

  
  print(df.head())

  
  df.to_csv(f"{ticker}_stock_data.csv")

  #Converting a complex Price to a more simple form
  from sklearn.preprocessing import MinMaxScaler
  scaler = MinMaxScaler()
  df['Close'] = scaler.fit_transform(df[['Close']])
  data = df[['Close']]
  print(df.columns)
  data.head()'''

import yfinance as yahoo
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from copy import deepcopy

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers


from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report


code = input()
#facebook = yahoo.Ticker("PHOENIXLTD.NS")
stock_data = yahoo.Ticker(code)
fb = stock_data.history(period="max").reset_index()
fb.index = pd.to_datetime(fb['Date'])  # Ensure proper datetime index
fb.index = fb.index.tz_localize(None)  # Remove timezone info

scaler = MinMaxScaler()
fb['Close'] = scaler.fit_transform(fb[['Close']])

fb = fb[['Close']]
print(fb.columns)
fb


fb.index = fb.index.tz_localize(None)


def fb_to_windowed_fb(dataframe, first_date, last_date, n=3):

  first_date = pd.to_datetime(first_date)
  last_date = pd.to_datetime(last_date)

  target_date = first_date

  dates = []
  X, Y = [], []

  last_time = False
  while True:
    fb_subset = dataframe.loc[:target_date].tail(n+1)

    if len(fb_subset) != n+1:
      print(f'Error: Window of size {n} is too large for date {target_date}')
      return

    values = fb_subset['Close'].to_numpy()
    x, y = values[:-1], values[-1]

    dates.append(target_date)
    X.append(x)
    Y.append(y)

    next_week = dataframe.loc[target_date:target_date+datetime.timedelta(days=7)]
    next_datetime_str = str(next_week.head(2).tail(1).index.values[0])
    next_date_str = next_datetime_str.split('T')[0]
    year_month_day = next_date_str.split('-')
    year, month, day = year_month_day
    next_date = datetime.datetime(day=int(day), month=int(month), year=int(year))

    if last_time:
      break

    target_date = next_date

    if target_date == last_date:
      last_time = True

  ret_fb = pd.DataFrame({})
  ret_fb['Target Date'] = dates

  X = np.array(X)
  for i in range(0, n):
    X[:, i]
    ret_fb[f'Target-{n-i}'] = X[:, i]

  ret_fb['Target'] = Y

  return ret_fb

windowed_fb = fb_to_windowed_fb(fb,
                                #'2019-04-05',
                                fb.index[15],
                                '2025-03-28',
                                n=15)
windowed_fb


def windowed_fb_to_date_X_y(windowed_dataframe):
  fb_as_np = windowed_dataframe.to_numpy()

  dates = fb_as_np[:, 0]

  middle_matrix = fb_as_np[:, 1:-1]
  X = middle_matrix.reshape((len(dates), middle_matrix.shape[1], 1))

  Y = fb_as_np[:, -1]

  return dates, X.astype(np.float32), Y.astype(np.float32)

dates, X, y = windowed_fb_to_date_X_y(windowed_fb)

dates.shape, X.shape, y.shape

model = Sequential([layers.Input((15, 1)),
                    layers.LSTM(64),
                    layers.Dense(32, activation='relu'),
                    layers.Dense(32, activation='relu'),
                    layers.Dense(1)])

model.compile(loss='mse',
              optimizer=Adam(learning_rate=0.001),
              metrics=['mean_absolute_error'])

model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=120)


#train_predictions = model.predict(X_train).flatten()

#plt.plot(dates_train, train_predictions)
#plt.plot(dates_train, y_train)
#plt.legend(['Training Predictions', 'Training Observations'])

# STEP 3: Inverse transform predictions and actuals
train_predictions = model.predict(X_train)
train_predictions = scaler.inverse_transform(train_predictions).flatten()

y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()

# STEP 4: Plotting predictions vs actuals
plt.figure(figsize=(300, 100))
plt1 = plt.subplot2grid((25, 20), (0, 0))
plt2 = plt.subplot2grid((25, 20), (1, 0))

plt1.plot(dates_train[1000:5000], y_train_actual[1000:5000],label='Training Observations')
plt2.plot(dates_train[1000:5000], train_predictions[1000:5000],label='Training Predictions', color="red")
plt1.legend()
plt2.legend()
plt1.set_title("Actual Stock Prices")
plt2.set_title("Predicted Stock Prices")
plt1.set_xlabel("Date")
plt1.set_ylabel("Close Price")
plt2.set_xlabel("Date")
plt2.set_ylabel("Close Price")
plt1.grid(True)
plt2.grid(True)
#plt.suptitle("Actual vs Predicted")
plt1 = plt.subplots_adjust(bottom=0.1,wspace=0.3,hspace=0.3)
plt.show()


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Example actual and predicted values
y_true = np.array(y_train_actual)
y_pred = np.array(train_predictions)

# Calculate MSE, RMSE, MAE, R2
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
mape = 100 * np.mean(abs((y_true - y_pred) / y_true), axis=-1)
r2 = r2_score(y_true, y_pred)

print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"MAPE: {mape}")
print(f"R2 Score: {r2}")

#val_predictions = model.predict(X_val).flatten()

#plt.plot(dates_val, val_predictions)
#plt.plot(dates_val, y_val)
#plt.legend(['Validation Predictions', 'Validation Observations'])


# for validation: Inverse transform predictions and targets
val_predictions = model.predict(X_val)
val_predictions = scaler.inverse_transform(val_predictions).flatten()

y_val_actual = scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()

# STEP 4 (for validation): Plot predictions vs actuals
plt.figure(figsize=(12, 6))
plt.plot(dates_val, val_predictions, label='Validation Predictions')
plt.plot(dates_val, y_val_actual, label='Validation Observations')
plt.legend()
plt.title("Validation: Model Predictions vs Actual Stock Prices")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.grid(True)
plt.show()

#plt.plot(dates_train, train_predictions)
#plt.plot(dates_train, y_train)
#plt.plot(dates_val, val_predictions)
#plt.plot(dates_val, y_val)
#plt.plot(dates_test, test_predictions)
#plt.plot(dates_test, y_test)
#plt.legend(['Training Predictions',
            #'Training Observations',
            #'Validation Predictions',
            #'Validation Observations',
            #'Testing Predictions',
            #'Testing Observations'])

# Inverse transform all predictions and actuals
train_predictions = scaler.inverse_transform(model.predict(X_train)).flatten()
y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()

val_predictions = scaler.inverse_transform(model.predict(X_val)).flatten()
y_val_actual = scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()

test_predictions = scaler.inverse_transform(model.predict(X_test)).flatten()
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

# Plot everything
plt.figure(figsize=(14, 6))
plt.plot(dates_train, train_predictions)
plt.plot(dates_train, y_train_actual)
plt.plot(dates_val, val_predictions)
plt.plot(dates_val, y_val_actual)
plt.plot(dates_test, test_predictions)
plt.plot(dates_test, y_test_actual)

plt.legend([
    'Training Predictions',
    'Training Observations',
    'Validation Predictions',
    'Validation Observations',
    'Testing Predictions',
    'Testing Observations'
])
plt.title("Full Timeline: Model Predictions vs Actual Stock Prices")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.grid(True)
plt.show()


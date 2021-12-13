import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mpl_dates
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor as dtr
from sklearn.linear_model import LinearRegression as lr


df = yf.download('ETH-USD', start='2021-07-12', end='2021-10-12', interval='1h')
df3 = pd.read_csv("ETHto2017_4.csv")
df3 = pd.DataFrame(df3)


df_close = df['Close']
#print(df_close[-1])
df_open= df['Open']
##print(df_open)
df_low = df['Low']
##print(df_low)
df_high = df['High']
##print(df_high)
df_dates = df.index.to_frame().reset_index(drop=True)


df['Date']=df_dates.values


ohlc= df.loc[:,['Date','Open','High','Low','Close']]
ohlc['Date'] = pd.to_datetime(ohlc['Date'])
ohlc['Date'] = ohlc['Date'].apply(mpl_dates.date2num)
ohlc = ohlc.astype(float)

# fig, ax = plt.subplots()

# candlestick_ohlc(ax, ohlc.values, width=0.6, colorup='green', colordown='red', alpha=0.8)

# ax.set_xlabel('Date')
# ax.set_ylabel('Price')

# fig.suptitle('Candlestick chart')

# date_format= mpl_dates.DateFormatter('%d-%m-%Y')
# ax.xaxis.set_major_formatter(date_format)
# fig.autofmt_xdate()
# fig.tight_layout()

df2=df['Close']
df2=pd.DataFrame(df2)

future_days = 500

df2['Pred'] = df2['Close'].shift(-future_days)

x = np.array(df2.drop(['Pred'],1))[:-future_days]
y = np.array(df2['Pred'])[:-future_days]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

tree = dtr().fit(x_train, y_train)
line = lr().fit(x_train,y_train)

x_future = df2.drop(['Pred'],1)[:-future_days]
x_future = x_future.tail(future_days)
x_future = np.array(x_future)

tree_prediction = tree.predict(x_future)
line_prediction = line.predict(x_future)

predictions = tree_prediction
valid = df2[x.shape[0]:]
valid['Pred'] = predictions

plt.figure(figsize=(16,8))
plt.xlabel('Days')
plt.ylabel('Close Price')
plt.plot(df2['Close'])
plt.plot(valid[['Close','Pred']])
plt.legend(['Original','Valid','Predicted'])

plt.show()


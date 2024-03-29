import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'

df = pd.read_excel("Superstore.xls")
furniture = df.loc[df['Category'] == 'Furniture']
#print(furniture['Order Date'].min(), furniture['Order Date'].max())
cols = ['Row ID', 'Order ID', 'Ship Date', 'Ship Mode',
'Customer ID', 'Customer Name', 'Segment', 'Country', 'City',
'State', 'Postal Code', 'Region', 'Product ID', 'Category',
'Sub-Category', 'Product Name', 'Quantity', 'Discount', 'Profit']

furniture.drop(cols, axis=1, inplace=True)
furniture = furniture.sort_values('Order Date')

furniture = furniture.groupby('Order Date')['Sales'].sum().reset_index()

furniture = furniture.set_index('Order Date')

y = furniture['Sales'].resample('MS').mean()

from pylab import rcParams
rcParams['figure.figsize'] = 18,8

# decomposition = sm.tsa.seasonal_decompose(y, model='additive')
# fig = decomposition.plot()
# plt.show()
# p = d = q = range(0, 2)
# pdq = list(itertools.product(p,d,q))
# seasonal_pdq = [(x[0], x[1], x[2], 12 ) for x in list(itertools.product(p,d,q))]

# for param in pdq:
	# for param_seasonal in seasonal_pdq:
		# try:
			# mod = sm.tsa.statespace.SARIMAX(y, order=param, seasonal_order=param_seasonal, enforce_stationarity=False, enforce_invertibility=False)

			# results = mod.fit()

			# print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal,results.aic))
		# except:
			# continue
			
mod = sm.tsa.statespace.SARIMAX(y, order=(1,1,1), seasonal_order=(1,1,0,12),enforce_stationarity=False,enforce_invertibility=False)
results= mod.fit()
print(results.summary().tables[1])

results.plot_diagnostics(figsize=(16, 8))
plt.show()

pred = results.get_prediction(start=pd.to_datetime('2017-01-01'), dynamic=False)
pred_ci = pred.conf_int()

ax = y['2014':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14,7))

ax.fill_between(pred_ci.index, pred_ci.iloc[:,0],pred_ci.iloc[:, 1], alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Furniture Sales')
plt.legend()

plt.show()

y_forecasted = pred.predicted_mean
y_truth = y['2017-01-01':]

mse = ((y_forecasted -y_truth) ** 2).mean()
print ('The Mean Squared Error of our forecasts is {}'.format(round(mse,2)))
print ('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))
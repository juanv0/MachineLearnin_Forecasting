#ForecastingModel Weekly Sales Transactions

#First Some data modelling, deleting P,W and truncating W0...W51, ands sales, to get columns=['Product_Code','Week','Sales'] 
import 	pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as sk
import sklearn.ensemble as en
from itertools import chain

df = pd.read_csv("Sales_Transactions_Dataset_Weekly.csv")
columns = df.columns[1:53].str.strip('W')
fm = pd.DataFrame(columns=['Product_Code','Week','Sales'])
product_Code = df['Product_Code'].str.strip('P').values

for c in columns:
	prod = product_Code
	week = [c]*811
	sales = df['W'+str(c)]
	d = {'Product_Code': prod, 'Week': week, 'Sales': sales}
	partial = pd.DataFrame(data=d)
	fm = fm.append(partial, ignore_index=True)

#plotting histogram (distribution of sales)
fm['Sales'].hist(bins=20)
plt.xlabel('Number of Sales')
plt.ylabel('Count Number of Sales')
plt.title('Sales')
plt.show()

#constuyendo la columna que contiene el one step back (1-lag)
#Building the column that has one step back
cfm = fm.iloc[:]
cfm['1_Week_Ago_Sales']=fm.loc[fm['Week'] != '51', ['Sales']]
#eliminando las filas que tienen valor Nan
#Droping rows that has Nan as value 
cfm=cfm.dropna()
#Droping rows from the frame that has value of week = 0
#Omitiendo todas las columnas del frame de la semana 0, pues solo consideramos las que tienen "antepasado"
fm = fm.drop(range(811))
fm=fm.reset_index()
#Reindexing
#Organizamos indices
#Mergin lag
fm['1_Week_Ago_Sales'] = cfm['1_Week_Ago_Sales'].loc[:]
#Creating np array that contains the diferentiation in sales per past week (w(i)-w(i-1))
deltasale = fm['Sales'].__array__().astype(float) - fm['1_Week_Ago_Sales'].__array__().astype(float)
#addin deltasale to the frame
fm['1_Week_Ago_Diff_1_Week_Ago_Sales'] = deltasale
#must delete created index
fm=fm.drop(['index'], axis=1)

#Lets make our Forward-Chaining Cross-Validation
#First we split in k=12 folds, then we make the base estimator wich
#assume that every sale will be the sames as the last week sale, later we will 
#calculate the score (error) of that aproximation, to latter on minimize the error
#we will minimeze the error using 
#1.Basic Feature Engeeniereing
#2.Statistical Transformation
#3.Random Forest
#4.LGBM Tunning
wi=40
wf=52
errors = []
for i in range(wi,wf):
	y_train = fm.loc[fm['Week']==str(i),['Sales']].values
	pred = fm.loc[fm['Week']==str(i),['1_Week_Ago_Sales']].values
	error = np.sqrt(sk.mean_squared_log_error(y_train,pred))
	errors.append(error)
	print ('error in fold {} = {:.3f}'.format(i,error))
	
print('Total error in base estimation {:.3f}'.format(np.mean(errors)))
#The above error are for the very basic Estimation next week sale will be the same, and we got an approximate value of %52 percent error
#wit rmsle

#Then calculate out wit estimator ---> RandomForestRegressor the prediccion, latter on we will calculate the rmsle error
errors = []
model = en.RandomForestRegressor(n_estimators=1000, n_jobs=-1, random_state=0)
fm["Week"]=pd.to_numeric(fm["Week"])
fm["Sales"]=pd.to_numeric(fm["Sales"])
def _train_test_split_time(X):
	for i in range(wi,wf):
		train = X[X["Week"]<i]
		val= X[X["Week"]==i]
		X_train, X_test = train.drop(["Sales"], axis=1), val.drop(["Sales"], axis=1)
		y_train, y_test = train["Sales"].values, val["Sales"].values
		yield X_train, X_test, y_train, y_test
		
def split(X):
	cv_t = _train_test_split_time(X)
	return chain(cv_t)

	# model.fit(x_train, y_train)
	# prediction = model.predict(x_test)
	# error = np.sqrt(sk.mean_squared_log_error(y_test, prediction))
	# errors.append(error)
	# print ('error in fold {} = {:.3f}'.format(i,error))
	
# print('Total error in base estimation {:.3f}'.format(np.mean(errors)))
errors_RFR=[]
	
for indx,fold in enumerate(split(fm)):
	X_train, X_test, y_train, y_test = fold
	model.fit(X_train, y_train)
	prediction = model.predict(X_test)
	error = sk.mean_squared_log_error(y_test, prediction)
	errors_RFR.append(error)
	dif = y_test - prediction
	print ('error in fold {} = {:.4f} '.format(indx,error))
	
print('Total Error {:.4f}'.format(np.mean(errors_RFR)))


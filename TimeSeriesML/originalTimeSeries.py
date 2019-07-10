#ForecastingModel Weekly Sales Transactions

#First Some data modelling, deleting P,W and truncating W0...W51, ands sales, to get columns=['Product_Code','Week','Sales'] 
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_log_error
from sklearn.ensemble import RandomForestRegressor

from sklearn import base

import warnings
warnings.filterwarnings('ignore')
import 	pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as sk
import sklearn.ensemble as en
from itertools import chain

# df = pd.read_csv("Sales_Transactions_Dataset_Weekly.csv")
# columns = df.columns[1:53].str.strip('W')
# df = pd.DataFrame(columns=['Product_Code','Week','Sales'])
# product_Code = df['Product_Code'].str.strip('P').values

# for c in columns:
	# prod = product_Code
	# week = [c]*811
	# sales = df['W'+str(c)]
	# d = {'Product_Code': prod, 'Week': week, 'Sales': sales}
	# partial = pd.DataFrame(data=d)
	# df = df.append(partial, ignore_index=True)

# # plotting histogram (distribution of sales)
# df['Sales'].hist(bins=20)
# plt.xlabel('Number of Sales')
# plt.ylabel('Count Number of Sales')
# plt.title('Sales')
# plt.show()

# # constuyendo la columna que contiene el one step back (1-lag)
# # Building the column that has one step back
# cdf = df.iloc[:]
# cdf['1_Week_Ago_Sales']=df.loc[df['Week'] != '51', ['Sales']]
# # eliminando las filas que tienen valor Nan
# # Droping rows that has Nan as value 
# cdf=cdf.dropna()
# # Droping rows from the frame that has value of week = 0
# # Omitiendo todas las columnas del frame de la semana 0, pues solo consideramos las que tienen "antepasado"
# df = df.drop(range(811))
# df=df.reset_index()
# # Reindexing
# # Organizamos indices
# # Mergin lag
# df['1_Week_Ago_Sales'] = cdf['1_Week_Ago_Sales'].loc[:]
# # Creating np array that contains the diferentiation in sales per past week (w(i)-w(i-1))
# deltasale = df['Sales'].__array__().astype(float) - df['1_Week_Ago_Sales'].__array__().astype(float)
# # addin deltasale to the frame
# df['1_Week_Ago_Diff_1_Week_Ago_Sales'] = deltasale
# # must delete created index
# df=df.drop(['index'], axis=1)
df_org = pd.read_csv('Sales_Transactions_Dataset_Weekly.csv')
df_org = df_org.filter(regex=r'Product|W')
df_org.head()
df = df_org.melt(id_vars='Product_Code', var_name='Week', value_name='Sales')

df['Product_Code'] = df['Product_Code'].str.extract('(\d+)', expand=False).astype(int)
df['Week'] = df['Week'].str.extract('(\d+)', expand=False).astype(int)

df = df.sort_values(['Week', 'Product_Code'])
df.head(10)
class ToSupervised(base.BaseEstimator,base.TransformerMixin):
    
    def __init__(self,col,groupCol,numLags,dropna=False):
        
        self.col = col
        self.groupCol = groupCol
        self.numLags = numLags
        self.dropna = dropna
        
    def fit(self,X,y=None):
        self.X = X
        return self
    
    def transform(self,X):
        tmp = self.X.copy()
        for i in range(1,self.numLags+1):
            tmp[str(i)+'_Week_Ago'+"_"+self.col] = tmp.groupby([self.groupCol])[self.col].shift(i) 
            
        if self.dropna:
            tmp = tmp.dropna()
            tmp = tmp.reset_index(drop=True)
            
        
            
        return tmp
		
class ToSupervisedDiff(base.BaseEstimator,base.TransformerMixin):
    
    def __init__(self,col,groupCol,numLags,dropna=False):
        
        self.col = col
        self.groupCol = groupCol
        self.numLags = numLags
        self.dropna = dropna
        
    def fit(self,X,y=None):
        self.X = X
        return self
    
    def transform(self,X):
        tmp = self.X.copy()
        for i in range(1,self.numLags+1):
            tmp[str(i)+'_Week_Ago_Diff_'+"_"+self.col] = tmp.groupby([self.groupCol])[self.col].diff(i) 
            
        if self.dropna:
            tmp = tmp.dropna()
            tmp = tmp.reset_index(drop=True)
            
        return tmp
		
from itertools import chain

class Kfold_time(object):
    
    def __init__(self,**options):
        
        
        self.target     = options.pop('target', None)
        self.date_col   = options.pop('date_col', None)
        self.date_init  = options.pop('date_init', None)
        self.date_final = options.pop('date_final', None)
        if options:
            raise TypeError("Invalid parameters passed: %s" %
                               str(options))
            
        if ((self.target==None )|(self.date_col==None )|
            (self.date_init==None )|(self.date_final==None )):
             
             raise TypeError("Incomplete inputs")
    
    def _train_test_split_time(self,X):
        n_arrays = len(X)
        if n_arrays == 0:
            raise ValueError("At least one array required as input")
        for i in range(self.date_init,self.date_final):
            train = X[X[self.date_col] < i]
            val   = X[X[self.date_col] == i]
            X_train, X_test = train.drop([self.target], axis=1), val.drop([self.target], axis=1)
            y_train, y_test = train[self.target].values, val[self.target].values
            yield X_train, X_test, y_train, y_test
    def split(self,X):
        cv_t = self._train_test_split_time(X)
        return chain(cv_t)

def rmsle(ytrue, ypred):
    return np.sqrt(mean_squared_log_error(ytrue, ypred))
		
class BaseEstimator(base.BaseEstimator, base.RegressorMixin):
    def __init__(self, predCol):
        """
            As a base model we assume the number of sales 
            last week and this week are the same
            Input: 
                    predCol: l-week ago sales
        """
        self.predCol = predCol
    def fit(self, X, y):
        return self
    def predict(self, X):
        prediction = X[self.predCol].values
        return prediction
    def score(self, X, y,scoring):
        
        prediction = self.predict(X)
    
        error =scoring(y, prediction)
        return error

class TimeSeriesRegressor(base.BaseEstimator, base.RegressorMixin):
    
    def __init__(self,model,cv,scoring,verbosity=True):
        self.model = model
        self.cv = cv
        self.verbosity = verbosity
        self.scoring = scoring 
        
            
    def fit(self,X,y=None):
        return self
        
    
    def predict(self,X=None):
        
        pred = {}
        for indx,fold in enumerate(self.cv.split(X)):

            X_train, X_test, y_train, y_test = fold    
            self.model.fit(X_train, y_train)
            pred[str(indx)+'_fold'] = self.model.predict(X_test)
            
        prediction = pd.DataFrame(pred)
    
        return prediction
    

    def score(self,X,y=None):


        errors = []
        for indx,fold in enumerate(self.cv.split(X)):

            X_train, X_test, y_train, y_test = fold    
            self.model.fit(X_train, y_train)
            prediction = self.model.predict(X_test)
            error = self.scoring(y_test, prediction)
            errors.append(error)

            if self.verbosity:
                print("Fold: {}, Error: {:.4f}".format(indx,error))

        if self.verbosity:
            print('Total Error {:.4f}'.format(np.mean(errors)))

        return errors


class TimeSeriesRegressorLog(base.BaseEstimator, base.RegressorMixin):
    
    def __init__(self,model,cv,scoring,verbosity=True):
        self.model = model
        self.cv = cv
        self.verbosity = verbosity
        self.scoring = scoring
        
            
    def fit(self,X,y=None):
        return self
        
    
    def predict(self,X=None):
        
        pred = {}
        for indx,fold in enumerate(self.cv.split(X)):

            X_train, X_test, y_train, y_test = fold    
            self.model.fit(X_train, y_train)
            pred[str(indx)+'_fold'] = self.model.predict(X_test)
            
        prediction = pd.DataFrame(pred)
    
        return prediction

    
    def score(self,X,y=None):#**options):


        errors = []
        for indx,fold in enumerate(self.cv.split(X)):

            X_train, X_test, y_train, y_test = fold    
            self.model.fit(X_train, np.log1p(y_train))
            prediction = np.expm1(self.model.predict(X_test))
            error = self.scoring(y_test, prediction)
            errors.append(error)

            if self.verbosity:
                print("Fold: {}, Error: {:.4f}".format(indx,error))

        if self.verbosity:
                print('Total Error {:.4f}'.format(np.mean(errors)))

        return errors
		
steps = [('1_step',
          ToSupervised('Sales','Product_Code',1)),
         ('1_step_diff',
          ToSupervisedDiff('1_Week_Ago_Sales',
          'Product_Code',1,dropna=True))]
super_1 = Pipeline(steps).fit_transform(df)

kf = Kfold_time(target='Sales',date_col = 'Week', 
                   date_init=40, date_final=52)

base_model = BaseEstimator('1_Week_Ago_Sales')
errors = []
for indx,fold in enumerate(kf.split(super_1)):
    X_train, X_test, y_train, y_test = fold
    error = base_model.score(X_test,y_test,rmsle)
    errors.append(error)
    print("Fold: {}, Error: {:.3f}".format(indx,error))
    
print('Total Error {:.3f}'.format(np.mean(errors)))
		
model = RandomForestRegressor(n_estimators=1000,
                               n_jobs=-1,
                                random_state=0)

steps_3 = [('1_step',ToSupervised('Sales','Product_Code',3)),
         ('1_step_diff',ToSupervisedDiff('1_Week_Ago_Sales','Product_Code',1)),
         ('2_step_diff',ToSupervisedDiff('2_Week_Ago_Sales','Product_Code',1)),
         ('3_step_diff',ToSupervisedDiff('3_Week_Ago_Sales','Product_Code',1,dropna=True)),
         ('predic_3',TimeSeriesRegressor(model=model,cv=kf,scoring=rmsle))]
super_3_p = Pipeline(steps_3).fit(df)		  

Model_3_Error = super_3_p.score(df)
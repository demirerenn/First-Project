
#installing required libraries

!pip install pandas
!pip install sklearn
!pip install missingno
!pip install ycimpute
!pip install xgboost
!pip install lightgbm
!pip install catboost


import numpy as np
import pandas as pd 
import missingno as msno 
import statsmodels.api as sm
import xgboost
conda install -c conda-forge lightgbm

#installing required functions
from ycimpute.imputer import knnimput
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale 
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import neighbors
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

df = pd.read_excel("Data.xlsx") #creating dataframe with pandas
print(df.head())


#change the gender class to numerical value

lbe=LabelEncoder()
lbe.fit_transform(df["Sex"])


#placing numerical values in new column

df["new_Sex"]=lbe.fit_transform(df["Sex"])
df.head()

#Identifying missing observations with the isnull function
df.isnull().sum()

#total number of missing observations
df.isnull().sum().sum()

#data missing at least one observation
df[df.isnull().any(axis=1)]

#data with complete observation information
df[df.notnull().all(axis=1)]

#Creating a data frame with numerical values only
df=df.loc[:,"Age":] 


#bar chart
print(msno.bar(df))

#matrix chart
msno.matrix(df)

#heatmap
msno.heatmap(df)

#fill in missing data with knn algorithms

#names of variables are saved in a list
var_names=list(df)
#dataframe is converted to numpy array
n_df=np.array(df)
n_df[0:10]
#Missing data are filled with KNN
dff=knnimput.KNN(k=8).complete(n_df)

dff=pd.DataFrame(dff,columns=var_names)
df2=pd.DataFrame(dff,columns=var_names)


clf = LocalOutlierFactor()
pred=clf.fit_predict(df2)
df2[pred==-1]
df2.drop([102], axis=0, inplace=True)
dff.isnull().sum()

dff.to_excel('./new_data.xlsx')


#multiple linear regression

y=dff["30m_Sprint_Time"]
dff=dff.drop("30m_Sprint_Time",axis=1)
X=dff.loc[:,"Age":] 

y.head()
X.head()

#Building a model with Statsmodels
lm=sm.OLS(y,X)
modelStats=lm.fit()
modelStats.summary()


#Building a model with scikit learn
lm  = LinearRegression()

model=lm.fit(X,y)
model.intercept_
model.coef_
X
type(X)
model.predict(X)

MSE = mean_squared_error(y, model.predict(X))
MSE

RMSE = np.sqrt(MSE)
RMSE

y.head()
model.predict(X)[0:4]

#test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 1)
X_train.head()
y_train.head()
X_test.head()
y_test.head()

lm = LinearRegression()
model = lm.fit(X_train, y_train)


#education error
np.sqrt(mean_squared_error(y_train, model.predict(X_train)))

#test error
np.sqrt(mean_squared_error(y_test, model.predict(X_test)))

#KNN(K-Nearest Neighbors)

#It makes predictions based on the similarities of observations.

X=pd.concat([X])
knn_model = KNeighborsRegressor().fit(X_train, y_train)

RMSE = []

for k in range(10):
    k = k+1
    knn_model = KNeighborsRegressor(n_neighbors = k).fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    RMSE.append(rmse)
    print("k=", k, "için RMSE değeri:", rmse)

#GridSearchCV is for automatically finding the proper k value
knn_params = {"n_neighbors": np.arange(1,30,1)}
knn = KNeighborsRegressor()
knn_cv_model = GridSearchCV(knn, knn_params, cv = 3).fit(X_train, y_train)
knn_cv_model.best_params_
knn_tuned = KNeighborsRegressor(n_neighbors =9).fit(X_train, y_train)
y_pred = knn_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

#SVR(support vector regression)

svr_model = SVR("linear") 
svr_params = {"C": [0.1,0.5,1,3]}
svr_cv_model = GridSearchCV(svr_model, svr_params, cv = 5).fit(X_train, y_train)
svr_cv_model.best_params_
svr_tuned = SVR("linear", C = svr_cv_model.best_params_["C"]).fit(X_train, y_train)
y_pred = svr_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

#Artificial neural networks

#homogenization-standardization
scaler = StandardScaler()

scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
scaler.fit(X_test)
X_test_scaled = scaler.transform(X_test)
mlp_model = MLPRegressor().fit(X_train_scaled, y_train)
mlp_model 

mlp_params = {"alpha": [0.1, 0.01, 0.02, 0.001, 0.0001], 
             "hidden_layer_sizes": [(10,20), (5,5), (100,100)]}

mlp_cv_model = GridSearchCV(mlp_model, mlp_params, cv = 10, verbose = 2, n_jobs = -1).fit(X_train_scaled, y_train)

mlp_cv_model.best_params_
mlp_tuned = MLPRegressor(alpha=0.1,hidden_layer_sizes=(100,100)).fit(X_train_scaled, y_train)
y_pred = mlp_tuned.predict(X_test_scaled)
np.sqrt(mean_squared_error(y_test, y_pred))

#CART (Classification and Regression Tree)

cart_params = {"max_depth": [2,3,4,5,10,20],
              "min_samples_split": [2,10,5,30,50,100]}
cart_model = DecisionTreeRegressor()
cart_cv_model = GridSearchCV(cart_model, cart_params, cv = 10).fit(X_train, y_train)
cart_cv_model.best_params_
cart_tuned = DecisionTreeRegressor(max_depth = 3, min_samples_split = 10).fit(X_train, y_train)
y_pred = cart_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

#Random Forests
rf_model = RandomForestRegressor(random_state = 42).fit(X_train, y_train)
rf_model

rf_params = {"max_depth": [5,8,10],
            "max_features": [2,5,10],
            "n_estimators": [200, 500, 1000, 2000],
            "min_samples_split": [2,10,80,100]}

rf_cv_model = GridSearchCV(rf_model, rf_params, cv = 10, n_jobs = -1, verbose = 2).fit(X_train, y_train)
rf_cv_model.best_params_
rf_model = RandomForestRegressor(random_state = 42, 
                                 max_depth = 8,
                                max_features = 5,
                                min_samples_split = 2,
                                 n_estimators = 200)
rf_tuned = rf_model.fit(X_train, y_train)
y_pred = rf_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

#variable impact level
rf_tuned.feature_importances_*100
Importance = pd.DataFrame({'Importance':rf_tuned.feature_importances_*100}, 
                          index = X_train.columns)


Importance.sort_values(by = 'Importance', 
                       axis = 0, 
                       ascending = True).plot(kind = 'barh', 
                                              color = 'r', )

plt.xlabel('Variable Importance')
plt.gca().legend_ = None

#Gradient Boosting Machines

gbm_model = GradientBoostingRegressor().fit(X_train, y_train)
gbm_params = {"learning_rate": [0.001,0.1,0.01],
             "max_depth": [3,5,8],
             "n_estimators": [100,200,500],
             "subsample": [1,0.5,0.8],
             "loss": ["ls","lad","quantile"]}
gbm_model = GradientBoostingRegressor().fit(X_train, y_train)
gbm_cv_model = GridSearchCV(gbm_model, 
                            gbm_params, 
                            cv = 10, 
                            n_jobs=-1, 
                            verbose = 2).fit(X_train, y_train)
gbm_cv_model.best_params_
gbm_tuned = GradientBoostingRegressor(learning_rate = 0.1,
                                     loss = "lad",
                                     max_depth = 3,
                                     n_estimators = 200,
                                     subsample = 0.5).fit(X_train, y_train)
y_pred = gbm_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

Importance = pd.DataFrame({'Importance':gbm_tuned.feature_importances_*100}, 
                          index = X_train.columns)


Importance.sort_values(by = 'Importance', 
                       axis = 0, 
                       ascending = True).plot(kind = 'barh', 
                                              color = 'r', )

plt.xlabel('Variable Importance')
plt.gca().legend_ = None

#XGBoost

xgb = XGBRegressor()
xgb_params = {"learning_rate": [0.1,0.01,0.5],
             "max_depth": [2,3,4,5,8],
             "n_estimators": [100,200,500,1000],
             "colsample_bytree": [0.4,0.7,1]}
xgb_cv_model  = GridSearchCV(xgb,xgb_params, cv = 10, n_jobs = -1, verbose = 2).fit(X_train, y_train)
xgb_cv_model.best_params_
xgb_tuned = XGBRegressor(colsample_bytree = 1, 
                         learning_rate = 0.01, 
                         max_depth = 2, 
                         n_estimators = 1000).fit(X_train, y_train)
y_pred = xgb_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

#LightGBM

lgb_model = LGBMRegressor()
lgbm_params = {"learning_rate": [0.01, 0.1, 0.5, 1],
              "n_estimators": [20,40,100,200,500,1000],
              "max_depth": [1,2,3,4,5,6,7,8,9,10]}

lgbm_cv_model = GridSearchCV(lgb_model, 
                             lgbm_params, 
                             cv = 10, 
                             n_jobs = -1, 
                             verbose =2).fit(X_train, y_train)
lgbm_cv_model.best_params_
lgbm_tuned = LGBMRegressor(learning_rate = 0.1, 
                          max_depth = 6, 
                          n_estimators = 20).fit(X_train, y_train)
y_pred = lgbm_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

#CatBoost

catb_model = CatBoostRegressor()
catb_params = {"iterations": [200,500,100],
              "learning_rate": [0.01,0.1],
              "depth": [3,6,8]}
catb_cv_model = GridSearchCV(catb_model, 
                           catb_params, 
                           cv = 5, 
                           n_jobs = -1, 
                           verbose = 2).fit(X_train, y_train)
catb_cv_model.best_params_
catb_tuned = CatBoostRegressor(depth = 3, iterations = 100, learning_rate = 0.1).fit(X_train, y_train)
y_pred = catb_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

df2
def compML(df2, y, alg):
    
    #train-test ayrimi
    y=df2["30m_Sprint_Time"]
    dff=df2.drop("30m_Sprint_Time",axis=1)
    X=dff.loc[:,"Age":] 
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=42)
    #modelleme
    model = alg().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
    model_ismi = alg.__name__
    print(model_ismi, "Modeli Test Hatası:",RMSE)
    
#Calculations are made with default values of the model.
models = [LGBMRegressor, 
          XGBRegressor, 
          GradientBoostingRegressor, 
          RandomForestRegressor, 
          DecisionTreeRegressor,
          MLPRegressor,
          KNeighborsRegressor, 
          SVR]

for i in models:
    compML(df2, "30m_Sprint_Time", i)


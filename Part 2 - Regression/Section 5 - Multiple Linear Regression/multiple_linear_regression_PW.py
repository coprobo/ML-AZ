# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding the Independent Variable
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
# '3' below tells the encoder which column we want to encode into a dummy variable
# since in our table, the 'State" is under 3rd column [0 indexed -> 0,1,2,3]
ct = ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)
# now our X has the states encoded into dummy data of 0s and 1s

# Avoiding the Dummy Variable TRAP
X = X[:, 1:] # we remove one dummy variable-> to avoid dummy variable trap



# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
# By applying the fit method onto the 'regressor' object, it means I fit
# the multiple Linear Regression to my training set
regressor.fit(X_train, y_train)



# Predicting the Test set results

# vector of predictions
y_pred = regressor.predict(X_test)


# Building the optimal model using Backward Elimination
import statsmodels.api as sm
# Multiple Linear Regression model is y =b0x0 + b1x1 + ... + bnxn
# where normally x0 is ommitted in the dataset because it is always =1
# however for 'Backward Elimination' using 'statsmodels' package
# we need to add that column of 1s to our data model, to mimic x0=1
X = np.append(arr = np.ones(shape=(len(X),1)).astype(int), values=X, axis =1)

# Backward Elimination
# X_opt will hold only statistically significant variables
# We start checking if the variables are significat first
# That;s why we need to import all the variables at the beginning
# which are 0,1,2,3,4,5 columns from X set
X_opt = X[:,[0,1,2,3,4,5]]

# 1. Select a significance level to stay in the model (e.g. SL=0.05)
# if a P-Value of an independent variable is above SL (P-Value > SL),
# then we will remove that variable, else it stays in the model

# 2. Fit the full model with all possible predictors
# OLS = Ordinary Least Squares method
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()

# 3. We print the summary tale to read which variable has the highest P-Value
regressor_OLS.summary()

# REPEAT ALL THE STEPS
X_opt = X[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
# Remove x1, which has the highest P-Value

# REPEAT ALL THE STEPS
X_opt = X[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
# Always look at 'Adj. R-squared' metric, which should be the highest in the
# selected model (closest to 1), if it starts dropping with the next step it means that 
# previous model was better fit

# REPEAT ALL THE STEPS
X_opt = X[:,[0,3,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary() # THIS IS THE BEST MODEL TO CHOOSE
# Always look at 'Adj. R-squared' metric, which should be the highest in the
# selected model (closest to 1), if it starts dropping with the next step it means that 
# previous model was better fit

# REPEAT ALL THE STEPS
X_opt = X[:,[0,3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
# Always look at 'Adj. R-squared' metric, which should be the highest in the
# selected model (closest to 1), if it starts dropping with the next step it means that 
# previous model was better fit


# How to interpret the 'COEF''
# 'coef' - coefficients of the model, if they are positive it means
# if the independent variable for that coefficient grows, the dependent variable
# will grow as well
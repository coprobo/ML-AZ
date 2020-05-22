# -*- coding: utf-8 -*-
"""
Created on Thu May 21 17:07:38 2020

@author: Piotr
"""


# Eclat - based on Apriori approach

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20) if pd.isnull(dataset.iloc[i,j]) == False])

# Training Eclat on the dataset
from apyori import apriori
rules = apriori(transactions = transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2)
# Visualising the results
results = list(rules)

# Generator objects put to nicely organised DataFrame
def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    return list(zip(lhs, rhs, supports))


# What is Support? support(Item1 and Item2) = [# transactions containing "Item1 and Item2" ] / [# transactions]
resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Product 1', 'Product 2', 'Support'])

resultsinDataFrame.nlargest(n = 10, columns = 'Support')
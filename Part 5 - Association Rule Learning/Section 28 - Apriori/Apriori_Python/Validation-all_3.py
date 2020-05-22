# MOST FREQUENT PARIS

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing - without NaNs (doesn't impact the data)
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20) if pd.isnull(dataset.iloc[i,j]) == False])
   
# UDEMY Data processing - with NaNs
# dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
# transactions = []
# for i in range(0, 7501):
#   transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])
    
# Most frequent pairs
from collections import Counter
from itertools import combinations

pairs = Counter()
for transaction in transactions:
    if len(transactions) < 2:
        continue
    transaction.sort()
    for combination in combinations(transaction, 2):
        pairs[combination] += 1
        
# 50 Most common pairs
print(pairs.most_common(10))



##############################################################################
# Eclat - based on Apriori approach

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



##############################################################################
# Apriori

# Data Preprocessing
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20) if pd.isnull(dataset.iloc[i,j]) == False])

# Training Apriori on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2)

# Visualising the results
results = list(rules)

# Generator objects put to nicely organised DataFrame
def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))

# What is Support? support(Item) = [# transactions containing "Item" ] / [# transactions]
# What is Confidence? confidence(Item1 -> Item2) = [# transactions containinc Item1 and Item2] / [# transactions containing Item1]
# What is Lift? lift(Item1 -> Item2) = confidence(Item1 -> Item2) / support(Item)

resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])

resultsinDataFrame.nlargest(n = 10, columns = 'Lift')

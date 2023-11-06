import numpy as np
import pandas as pd 
import os
import matplotlib.pyplot as plt
from lib.models import *
from lib.configs import datasets, countries, estimator_vals, logg_every_e
from datetime import datetime
import time

start_time = time.time()

df = pd.read_csv('crosslearning/data/owid-covid-data.csv')

def error_register(train, test, lambdas = None, constraint = None):
    vals = {}
    vals['train'] = train
    vals['test'] = test
    if lambdas != None:
        vals['lambdas'] = lambdas
    if constraint != None:
        vals['constraint'] = constraint
    return vals

logger = {}
estimator = {}

epsilonsParametric= np.arange(1/1000, 1/10, 2/100)
epsilonsFunctional = np.arange(1, 1000,100)

logger['epsilonsFunctional'] = epsilonsFunctional
logger['epsilonsParametric'] = epsilonsParametric

X = []
for country in countries:
    X = X + [[datasets[country]['train']['array'], datasets[country]['population']]]
print(f"Number of countries {len(X)}")
    

# 1. Start with the centralized estimator
estimator['centralized'] = estimatorCovid(**estimator_vals['centralized'])
estimator['centralized'].fitCentralized(X)
logger['centralized'] = {}
for country in countries:
    error_cent = estimator['centralized'].evaluate(datasets[country]['train']['array'])
    acc_cent = estimator['centralized'].evaluate(datasets[country]['test']['array'])
    logger['centralized'][country] = error_register(error_cent, acc_cent)

# 2. Continue with independent 
logger['independent'] = {}
for country in countries:
    estimator[country] = estimatorCovid(**estimator_vals[country] ) 
    x_0 = datasets[country]['train']['array'][:,0]
    estimator[country].fitIndependent(datasets[country]['train']['array'])
    thisError = estimator[country].evaluate(datasets[country]['train']['array'])
    thisAcc = estimator[country].evaluate(datasets[country]['test']['array'])
    logger['independent'][country] = error_register(thisError, thisAcc)

# 3. Continue with Functional Constraints
logger['CLFunctional'] = {}
for epsilon in epsilonsFunctional:
    estimator['CLFunctional'] = estimatorCovid(**estimator_vals['CLParametric'])
    estimator['CLFunctional'].fitFunctional(X, epsilon)
    logger['CLFunctional'][str(epsilon)] = {}
    for idx, country in enumerate(countries):
        thisError = estimator['CLFunctional'].evaluate(datasets[country]['train']['array'])
        thisAcc = estimator['CLFunctional'].evaluate(datasets[country]['test']['array'])
        logger['CLFunctional'][str(epsilon)][country] = error_register(thisError, thisAcc, estimator['CLFunctional'].logger[idx]['lambdas'],estimator['CLFunctional'].logger[idx]['constraints'])

# 4. Continue with Parametric Constraints
logger['CLParametric'] = {}
for epsilon in epsilonsParametric:
    estimator['CLParametric'] = estimatorCovid(**estimator_vals['CLParametric'])
    estimator['CLParametric'].fitParametric(X, epsilon)
    logger['CLParametric'][str(epsilon)] = {}
    for country in countries:
        thisError = estimator['CLParametric'].evaluate(datasets[country]['train']['array'])
        thisAcc = estimator['CLParametric'].evaluate(datasets[country]['test']['array'])
        logger['CLParametric'][str(epsilon)][country] = error_register(thisError, thisAcc)

# Get the current date and time
current_datetime = datetime.now()

# Format the date and time as a string
formatted_datetime = current_datetime.strftime("%Y-%m-%d%H:%M:%S")

file_path = os.getcwd()+"/crosslearning/output/"+formatted_datetime+"_data.pkl"
import pickle
print(file_path)
with open(file_path, "wb") as pickle_file:
    pickle.dump(logger, pickle_file)

end_time = time.time()
execution_time = end_time - start_time

print(f"Execution time: {execution_time:.2f} seconds")
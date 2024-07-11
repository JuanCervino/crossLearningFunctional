import numpy as np
import pandas as pd 
import os
import matplotlib.pyplot as plt
from lib.models import *
from lib.configs import datasets, countries, estimator_vals, logg_every_e
from datetime import datetime
import time
import pickle

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

# epsilonsParametric= np.arange(1/100, 1/100, 2/10)
# epsilonsFunctional = np.arange(10000, 10000,1000)


epsilonsParametric= [0.0001, 0.001, 0.01, 0.1]

# epsilonsParametric= [0.001]
epsilonsFunctional = [0.1]
# epsilonsFunctional = []


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
    estimator['CLFunctional'] = estimatorCovid(**estimator_vals['CLFunctional'])
    estimator['CLFunctional'].fitFunctional(X, epsilon)
    logger['CLFunctional'][str(epsilon)] = {}
    for idx, country in enumerate(countries):
        thisError = estimator['CLFunctional'].evaluate(datasets[country]['train']['array'])
        thisAcc = estimator['CLFunctional'].evaluate(datasets[country]['test']['array'])
        logger['CLFunctional'][str(epsilon)][country] = error_register(thisError, thisAcc, estimator['CLFunctional'].logger[idx]['lambdas'],estimator['CLFunctional'].logger[idx]['constraints'])

# 4. Continue with Parametric Constraints
logger['CLParametric'] = {}
for epsilon in epsilonsParametric:
    if epsilon <= 0.05:
        estimator['CLParametric'] = estimatorCovid(**estimator_vals['CLParametricSmall'])
    else:
        estimator['CLParametric'] = estimatorCovid(**estimator_vals['CLParametric'])
    estimator['CLParametric'].fitParametric(X, epsilon)
    logger['CLParametric'][str(epsilon)] = {}
    for country in countries:
        thisError = estimator['CLParametric'].evaluate(datasets[country]['train']['array'])
        thisAcc = estimator['CLParametric'].evaluate(datasets[country]['test']['array'])
        logger['CLParametric'][str(epsilon)][country] = error_register(thisError, thisAcc)
        logger['CLParametric'][str(epsilon)]['beta'] =  [estimator['CLParametric'].betaIndependent,estimator['CLParametric'].betaCentral]      
        logger['CLParametric'][str(epsilon)]['gamma'] =  [estimator['CLParametric'].gammaIndependent,estimator['CLParametric'].gammaCentral]                   
             
# Get the current date and time
current_datetime = datetime.now()

# Format the date and time as a string
formatted_datetime = current_datetime.strftime("%Y-%m-%d%H:%M:%S")

file_path = os.getcwd()+"/crosslearning/output/"+formatted_datetime+"_data.pkl"
print(file_path)
with open(file_path, "wb") as pickle_file:
    pickle.dump(logger, pickle_file)

end_time = time.time()
execution_time = end_time - start_time

print(f"Execution time: {execution_time:.2f} seconds")


with open(file_path, "rb") as pickle_file:
    loaded_data = pickle.load(pickle_file)

trainAcc = 0
testAcc = 0

for country in countries:
    print(f"Country {country}")
    print(f"For Centralized train error {loaded_data['centralized'][country]['train']} and test error {loaded_data['centralized'][country]['test']}" )
    print(f"For Independent train error {loaded_data['independent'][country]['train']} and test error {loaded_data['independent'][country]['test']}" )
    # print('Parametric')
    for epsilon in loaded_data['epsilonsParametric']:
        print(f"For Parametric with epsilon {epsilon} train error {loaded_data['CLParametric'][str(epsilon)][country]['train']} and test error {loaded_data['CLParametric'][str(epsilon)][country]['test']}" )
    # print('Functional')
    for epsilon in loaded_data['epsilonsFunctional']:
        print(f"For Functional with epsilon {epsilon} train error {loaded_data['CLFunctional'][str(epsilon)][country]['train']} and test error {loaded_data['CLFunctional'][str(epsilon)][country]['test']}" )
        print(f"Functional error {loaded_data['CLFunctional'][str(epsilon)][country]['constraint'][-1]}")

print(f"For Functional {epsilon} train error {loaded_data['CLFunctional'][str(epsilon)][country]['lambdas']} and test error {loaded_data['CLFunctional'][str(epsilon)][country]['constraint']}" )


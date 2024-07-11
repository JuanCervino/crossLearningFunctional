import pickle
from lib.models import *
from lib.configs import datasets, countries, estimator_vals, logg_every_e
import os
import matplotlib.pyplot as plt

# def addOverCountries(listOfCountries, errorDict):
#     errorDict [] 
#     return errorTrain, errorTest

file_path = os.getcwd()+"/crosslearning/output/2023-11-1309:55:28_data.pkl"

with open(file_path, "rb") as pickle_file:
    loaded_data = pickle.load(pickle_file)

trainAcc = 0
testAcc = 0

for country in countries:
    print(f"Country {country}")
    print(f"For Centralized train error {loaded_data['centralized'][country]['train']} and test error {loaded_data['centralized'][country]['test']}" )
    print(f"For Independent train error {loaded_data['independent'][country]['train']} and test error {loaded_data['independent'][country]['test']}" )
    print('Parametric')
    for epsilon in loaded_data['epsilonsParametric']:
        print(f"For Parametric with epsilon {epsilon} train error {loaded_data['CLParametric'][str(epsilon)][country]['train']} and test error {loaded_data['CLParametric'][str(epsilon)][country]['test']}" )
    print('Functional')
    for epsilon in loaded_data['epsilonsFunctional']:
        print(f"For Functional with epsilon {epsilon} train error {loaded_data['CLFunctional'][str(epsilon)][country]['train']} and test error {loaded_data['CLFunctional'][str(epsilon)][country]['test']}" )

print(f"For Functional {epsilon} train error {loaded_data['CLFunctional'][str(epsilon)][country]['lambdas']} and test error {loaded_data['CLFunctional'][str(epsilon)][country]['constraint']}" )



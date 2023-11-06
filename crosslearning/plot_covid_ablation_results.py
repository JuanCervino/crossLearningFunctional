import pickle
from lib.models import *
from lib.configs import datasets, countries, estimator_vals, logg_every_e
import os
import matplotlib.pyplot as plt

# def addOverCountries(listOfCountries, errorDict):
#     errorDict [] 
#     return errorTrain, errorTest

file_path = os.getcwd()+"/crosslearning/output/2023-11-0618:08:52_data.pkl"

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
        print(f"For Parametric {epsilon} train error {loaded_data['CLParametric'][str(epsilon)][country]['train']} and test error {loaded_data['CLParametric'][str(epsilon)][country]['test']}" )
    for epsilon in loaded_data['epsilonsFunctional']:
        print(f"For Functional {epsilon} train error {loaded_data['CLFunctional'][str(epsilon)][country]['train']} and test error {loaded_data['CLFunctional'][str(epsilon)][country]['test']}" )

print(f"For Functional {epsilon} train error {loaded_data['CLFunctional'][str(epsilon)][country]['lambdas']} and test error {loaded_data['CLFunctional'][str(epsilon)][country]['constraint']}" )


# for key in loaded_data.keys():
#     for country in countries:
#     if key == 'centralized':
#         print(f"For {key} train error {loaded_data[key]['train']} and test error {loaded_data[key]['test']}" )
#     elif key == 'CLFunctional':
#         for epsilons in loaded_data['epsilonsFunctional']:
#             print(f" For Functional {epsilons} train {loaded_data[key][str(epsilons)]['train']} test {loaded_data[key][str(epsilons)]['test']}")
#     elif key == 'CLParametric':
#         for epsilons in loaded_data['epsilonsParametric']:
#             print(f" For Parametric {epsilons} train {loaded_data[key][str(epsilons)]['train']} test {loaded_data[key][str(epsilons)]['test']}")
    

#     elif key not in  ['epsilonsFunctional','epsilonsParametric']:
#         trainAcc += loaded_data[key]['train']
#         testAcc += loaded_data[key]['test']
# print(f" For Independent train {trainAcc} test {testAcc}")


# plt.hist(exam_scores, bins=10, edgecolor='k', alpha=0.7)
# plt.xlabel('Exam Scores')
# plt.ylabel('Frequency')
# plt.title('Exam Scores Histogram')
# plt.show()
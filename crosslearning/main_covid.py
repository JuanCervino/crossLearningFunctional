import numpy as np
import pandas as pd 
import os
import matplotlib.pyplot as plt
from lib.models import *
from lib.configs import datasets, countries, estimator_vals,logg_every_e

df = pd.read_csv('crosslearning/data/owid-covid-data.csv')


error = {}
acc = {}

estimator = {}

X = []
# countries = countries[:-4]
for country in countries:
    X = X + [[datasets[country]['train']['array'], datasets[country]['population']]]
print(len(X))
estimator['centralized'] = estimatorCovid(**estimator_vals['centralized'])
estimator['centralized'].fitCentralized(X)
estimator['CLParametric'] = estimatorCovid(**estimator_vals['CLParametric'])
estimator['CLParametric'].fitParametric(X, 0.05)
estimator['CLFunctional'] = estimatorCovid(**estimator_vals['CLParametric'])
estimator['CLFunctional'].fitFunctional(X, 80)

for country in countries:
    
    estimator[country] = estimatorCovid(**estimator_vals[country] )

    x_0 = datasets[country]['train']['array'][:,0]

    estimator[country].fitIndependent(datasets[country]['train']['array'])
    print(country, estimator[country].beta, estimator[country].gamma)
    error[country] = estimator[country].evaluate(datasets[country]['train']['array'])
    acc[country] = estimator[country].evaluate(datasets[country]['test']['array'])
    s, i ,r = estimator[country].getRolloutMatrix(x_0,datasets[country]['train']['array'].shape[1] )
    sc, ic ,rc = estimator['centralized'].getRolloutMatrix(x_0,datasets[country]['train']['array'].shape[1] )

print('beta Parametric',estimator['CLParametric'].betaIndependent)
print('gamma Parametric',estimator['CLParametric'].gammaIndependent)

print('beta Functional',estimator['CLParametric'].betaIndependent)
print('gamma Functional',estimator['CLParametric'].gammaIndependent)

print('Train')
for idx, country in enumerate(countries):
    error_cent = estimator['centralized'].evaluate(datasets[country]['train']['array'])
    error_cl_param = estimator['CLParametric'].evaluateIndependent(datasets[country]['train']['array'],datasets[country]['population'])
    error_cl_func = estimator['CLFunctional'].evaluateIndependent(datasets[country]['train']['array'],datasets[country]['population'])

    print(country, f"{error[country]:.2f}", f"{error_cent:.2f}", f"{error_cl_param[idx]:.2f}", f"{error_cl_func[idx]:.2f}")


print('Test')
for idx, country in enumerate(countries):
    acc_cent = estimator['centralized'].evaluate(datasets[country]['test']['array'])
    acc_cl_param = estimator['CLParametric'].evaluateIndependent(datasets[country]['test']['array'],datasets[country]['population'])
    acc_cl_func = estimator['CLFunctional'].evaluateIndependent(datasets[country]['test']['array'],datasets[country]['population'])

    print(country, f"{acc[country]:.2f}", f"{acc_cent:.2f}", f"{acc_cl_param[idx]:.2f}", f"{acc_cl_func[idx]:.2f}")

print('Totals')




# Now Plot them
colorlist =plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.figure()
for idx, country in enumerate(countries):
    print(estimator[country].beta, estimator[country].gamma)
    plt.plot(estimator[country].beta, estimator[country].gamma,'*',label=country+' idp', color=colorlist[idx])
    plt.plot(estimator['CLParametric'].betaIndependent[idx],estimator['CLParametric'].gammaIndependent[idx],'o',label=country+' cl param', color=colorlist[idx])
    plt.plot(estimator['CLFunctional'].betaIndependent[idx],estimator['CLFunctional'].gammaIndependent[idx],'^',label=country+' cl fun', color=colorlist[idx])

plt.legend(ncol=3)
plt.savefig('figs.pdf')

plt.figure(figsize=(12, 4))
fig, axs = plt.subplots(2)

for idx, country in enumerate(countries):
    x_vec = logg_every_e*np.arange(len(estimator['CLFunctional'].logger[idx]['lambdas']))
    axs[0].plot(x_vec,estimator['CLFunctional'].logger[idx]['lambdas'], color=colorlist[idx] )
    axs[1].plot(x_vec,estimator['CLFunctional'].logger[idx]['constraints'], color=colorlist[idx] )

axs[0].set_title('Dual Variable')
axs[1].set_title('Constraint Satisfaction')

# Adjust the layout to place the figures side by side
plt.subplots_adjust(wspace=0.5)  # Adjust the horizontal space between figures

# Show the figures
plt.savefig('duals.pdf')

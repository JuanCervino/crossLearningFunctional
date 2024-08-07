from lib.utils import *

countries = [ 'URY','USA', 'ESP', 'ARG','BRA','MEX','PRY', 'ITA']

start = 77
mid = 110
stop = 117
datasets = get_SIR_covid_datasets(countries, start, mid, stop)
epochs = 5000
logg_every_e = 200
eta_dual = 1e1000

estimator_vals = {}
estimator_vals['URY'] = {'epochs' : epochs,
                    'beta' : 1e-3/ datasets['URY']['population'],
                    'gamma' : 2e-3,
                    'T' : 1,
                    'population' : datasets['URY']['population'],
                    'eta' : 1e-4/ datasets['URY']['population'],
                    'eta_cent' : [0],
                    'eta_dual' : eta_dual,
                    'logging' : False, 
                    'logg_every_e' : logg_every_e,
                    }
estimator_vals['USA'] = {'epochs' : epochs,
                    'beta' : 1e-10/ datasets['USA']['population'],
                    'gamma' : 2e-3,
                    'T' : 1,
                    'population' : datasets['USA']['population'],
                    'eta' : 1e-8/ datasets['USA']['population'],
                    'eta_cent' : [0],
                    'eta_dual' : eta_dual,
                    'logging' : False, 
                    'logg_every_e' : logg_every_e,
                    }
estimator_vals['ESP'] = {'epochs' : epochs,
                    'beta' : 1e-3/ datasets['ESP']['population'],
                    'gamma' : 2e-3,
                    'T' : 1,
                    'population' : datasets['ESP']['population'],
                    'eta' : 1e-7/ datasets['ESP']['population'],
                    'eta_cent' : [0],                    
                    'eta_dual' : eta_dual,
                    'logging' : False, 
                    'logg_every_e' : logg_every_e,
                    }
estimator_vals['ARG'] = {'epochs' : epochs,
                    'beta' : 1e-3/ datasets['ARG']['population'],
                    'gamma' : 2e-3,
                    'T' : 1,
                    'population' : datasets['ARG']['population'],
                    'eta' : 1e-4/ datasets['ARG']['population'],
                    'eta_cent' : [0],                 
                    'eta_dual' : eta_dual,
                    'logging' : False, 
                    'logg_every_e' : logg_every_e,
                    }
estimator_vals['BRA'] = {'epochs' : epochs,
                    'beta' : 1e-3/ datasets['BRA']['population'],
                    'gamma' : 2e-3,
                    'T' : 1,
                    'population' : datasets['BRA']['population'],
                    'eta' : 1e-4/ datasets['BRA']['population'],
                    'eta_cent' : [0],                    
                    'eta_dual' : eta_dual,
                    'logging' : False, 
                    'logg_every_e' : logg_every_e,
                    }
estimator_vals['MEX'] = {'epochs' : epochs,
                    'beta' : 1e-3/ datasets['MEX']['population'],
                    'gamma' : 2e-3,
                    'T' : 1,
                    'population' : datasets['MEX']['population'],
                    'eta' : 1e-4/ datasets['MEX']['population'],
                    'eta_cent' : [0],
                    'eta_dual' : eta_dual,
                    'logging' : False, 
                    'logg_every_e' : logg_every_e,
                    }
estimator_vals['PRY'] = {'epochs' : epochs,
                    'beta' : 1e-2/ datasets['PRY']['population'],
                    'gamma' : 2e-3,
                    'T' : 1,
                    'population' : datasets['PRY']['population'],
                    'eta' : 1e-2/ datasets['PRY']['population'],
                    'eta_cent' : [0],                    
                    'eta_dual' : eta_dual,
                    'logging' : False, 
                    'logg_every_e' : logg_every_e,
                    }
estimator_vals['ITA'] = {'epochs' : epochs,
                    'beta' : 1e-7/ datasets['ITA']['population'],
                    'gamma' : 2e-3,
                    'T' : 1,
                    'population' : datasets['ITA']['population'],
                    'eta' : 1e-6/ datasets['ITA']['population'],
                    'eta_cent' : [0],                    
                    'eta_dual' : eta_dual,
                    'logging' : False, 
                    'logg_every_e' : logg_every_e,
                    }
ls_of_eta = [estimator_vals[key]['eta'] for key in estimator_vals.keys()]
estimator_vals['centralized'] = {'epochs' : epochs,
                    'beta' : 1e-10,
                    'gamma' : 2e-3,
                    'T' : 1,
                    'population' : 1,
                    # 'eta' : 1e-15,
                    'eta' : 1e-18,
                    'eta_cent' : ls_of_eta,
                    'eta_dual' : eta_dual,
                    'logging' : False, 
                    'logg_every_e' : logg_every_e,
                    }
estimator_vals['CLParametricSmall'] = {'epochs' : epochs,
                    'beta' : 1e-10,
                    'gamma' : 2e-3,
                    'T' : 1,
                    'population' : 1,
                    'eta' : 1e-15,
                    'eta_cent' : ls_of_eta,
                    # 'eta_dual' : 1e-3,
                    'eta_dual' : 1e0, # for smaller epsilons
                    # 'eta_dual' : 1e1,  # for larger epsilons
                    'logging' : True, 
                    'logg_every_e' : logg_every_e,
                    }
estimator_vals['CLParametric'] = {'epochs' : epochs,
                    'beta' : 1e-10,
                    'gamma' : 2e-3,
                    'T' : 1,
                    'population' : 1,
                    'eta' : 1e-15,
                    'eta_cent' : ls_of_eta,
                    # 'eta_dual' : 1e-3,
                    # 'eta_dual' : 1e2, # for smaller epsilons
                    'eta_dual' : 1e1,  # for larger epsilons
                    'logging' : True, 
                    'logg_every_e' : logg_every_e,
                    }
eta = 1e-18
estimator_vals['CLFunctional'] = {'epochs' :  logg_every_e*epochs,
                    'beta' : 1e-10,
                    'gamma' : 2e-3,
                    'T' : 1,
                    'population' : 1,
                    'eta' : eta,
                    'eta_cent' : [eta for key in estimator_vals.keys()],
                    'eta_dual' : 1e1,
                    'logging' : True, 
                    'logg_every_e' : logg_every_e,
                    }
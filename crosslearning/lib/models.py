import numpy as np
from dataclasses import dataclass

@dataclass
class estimatorCovid:
    
    epochs : int 
    beta : 1
    gamma : 2
    population : 1 
    T : 1
    eta : 1e-6
    eta_cent : [0]
    eta_dual : 1e-6
    logging : False 
    logg_every_e : 100

    def getUpdate(self,x):
        # x is a vector (3,1)
        return x + np.array([-self.beta*x[0]*x[1]*self.T/self.population,
                              self.beta*x[0]*x[1]/self.population-self.gamma*x[1],
                              self.gamma*x[1]])
    
    def getRolloutMatrix(self, x_0, steps):
        rollout = np.zeros((3,steps))
        rollout[:,0] = x_0

        for i in range(1,steps):
            rollout[:,i] = self.getUpdate(rollout[:,i-1])

        return rollout

    def getGradient(self, roll, X):
        nabla = np.zeros(2,) # beta, gamma
        delta = roll[:, 1:] - X[:, 1:] 
        # print('delta', delta.shape,roll.shape)
        dSdBeta = -(self.T/self.population)*np.multiply(X[0,:-1],X[1,:-1])
        dRdGamma = X[1,1:]

        nabla[0] = np.sum(np.multiply(delta[0,:], dSdBeta) )
        nabla[1] = np.sum(np.multiply(delta[2,:], dRdGamma) )
        return nabla

    def fitIndependent(self, X):
        # X is a matrix (3,n)
        # S_{t+1} = S_{t} - T \beta I S / N T
        # I_{t+1} = I_{t} +\beta I S T / N - \gamma I
        # S_{t+1} = S_t + \gamma I_t T
        # R_0 = \beta / \gamma
        
        for e in range(self.epochs):
            x_0 = X[:,0]
            roll = self.getRolloutMatrix(x_0, X.shape[1])

            nablaBeta, nablaGamma = self.getGradient(roll, X)
            self.beta = self.beta - self.eta * nablaBeta
            self.gamma = self.gamma - self.eta * nablaGamma
            # print(self.beta, self.gamma)
        pass
    
    def fitCentralized(self, datasets):
        # X is a matrix (3,n)
        # S_{t+1} = S_{t} - T \beta I S / N T
        # I_{t+1} = I_{t} +\beta I S T / N - \gamma I
        # S_{t+1} = S_t + \gamma I_t T
        # R_0 = \beta / \gamma
        
        for e in range(self.epochs):
            nablaBetaAcc, nablaGammaAcc = 0, 0
            for idx, thisDataset in enumerate(datasets):
                data, population = thisDataset
                self.population = population
                x_0 = data[:,0]
                roll = self.getRolloutMatrix(x_0, data.shape[1])

                nablaBeta, nablaGamma = self.getGradient(roll, data)
                nablaBetaAcc, nablaGammaAcc = nablaBetaAcc+nablaBeta, nablaGammaAcc+nablaGamma

            self.beta = self.beta - self.eta * nablaBetaAcc
            self.gamma = self.gamma - self.eta * nablaGammaAcc
            # print(self.beta, self.gamma)
        pass

    def createEstimators(self, betaInit, gammaInit, numberOfEstimators):
        self.betaIndependent = [betaInit for i in range(numberOfEstimators)]
        self.gammaIndependent = [gammaInit for i in range(numberOfEstimators)]
        self.betaCentral = betaInit
        self.gammaCentral = gammaInit
        pass
    
    def getLambdaA(self, mus):
        lambdas = np.divide(mus,mus+1)
        a = 1/np.linalg.norm(np.append(lambdas,1), 1)
        return lambdas, a

    def projectParametric(self):
        dualSteps = 100
        mus = np.zeros(len(self.betaIndependent))
        theta_g = np.array([self.betaCentral, self.gammaCentral])
        theta_i = np.array([self.betaIndependent,
                           self.gammaIndependent])
        lambdas, a = self.getLambdaA(mus)
        theta_g_aux = a*(theta_g+theta_i@lambdas)
        theta_i_aux = np.multiply(theta_i, (1-lambdas))
        + theta_g_aux[:,np.newaxis]@lambdas[np.newaxis,:]
        for i in range(dualSteps):
            # Update value of mu
            val = theta_i_aux-theta_g_aux[:,np.newaxis]
            # print(np.power(np.linalg.norm(val, axis = 0),2))
            mus = mus + self.eta_dual*(np.power(np.linalg.norm(val, axis = 0),2)
                                     -np.ones(len(self.betaIndependent))*self.epsilon**2)
            # Make it positive
            mus = np.maximum(mus,0)
            # Update Values
            lambdas, a = self.getLambdaA(mus)
            for i in range(len(self.betaIndependent)):
                theta_g_aux = a*(theta_g+theta_i@lambdas)
                theta_i_aux = np.multiply(theta_i, (1-lambdas))
                + theta_g_aux[:,np.newaxis]@lambdas[np.newaxis,:]
        # Assign values
        self.betaCentral = theta_g_aux[0]
        self.gammaCentral = theta_g_aux[1]
        for i in range(len(self.betaIndependent)):
            self.betaIndependent[i] = theta_i_aux[0,i]
            self.gammaIndependent[i] = theta_i_aux[1,i]
        pass

    def fitParametric(self, datasets, epsilon):
        # X is a matrix (3,n)
        # S_{t+1} = S_{t} - T \beta I S / N T
        # I_{t+1} = I_{t} +\beta I S T / N - \gamma I
        # S_{t+1} = S_t + \gamma I_t T
        # R_0 = \beta / \gamma
        self.betaIndependent = [self.beta for i in range(len(datasets))]
        self.gammaIndependent = [self.gamma for i in range(len(datasets))]
        self.betaCentral = self.beta
        self.gammaCentral = self.gamma
        self.epsilon = epsilon
        for e in range(self.epochs):
            for idx, thisDataset in enumerate(datasets):
                data, population = thisDataset
                # Compute the update
                self.population = population
                x_0 = data[:,0]
                self.beta = self.betaIndependent[idx] 
                self.gamma = self.gammaIndependent[idx] 
                roll = self.getRolloutMatrix(x_0, data.shape[1])
                nablaBeta, nablaGamma = self.getGradient(roll, data)
                self.betaIndependent[idx] = self.betaIndependent[idx] - self.eta_cent[idx] * nablaBeta
                self.gammaIndependent[idx] = self.gammaIndependent[idx] - self.eta_cent[idx] * nablaGamma

            self.projectParametric()
        pass

    def evaluateConstraint(self, datasets):
        constraintSlack = []
        for idx, thisDataset in enumerate(datasets):
            data, population = thisDataset
            # Compute the update
            self.population = population
            x_0 = data[:,0]
            self.beta = self.betaIndependent[idx] 
            self.gamma = self.gammaIndependent[idx] 
            roll = self.getRolloutMatrix(x_0, data.shape[1])
            self.beta = self.betaCentral
            self.gamma = self.gammaCentral
            rollCentral = self.getRolloutMatrix(x_0, data.shape[1])
            constraintSlack = constraintSlack + [np.linalg.norm(roll-rollCentral)**2/population]
        return constraintSlack

    def fitFunctional(self, datasets, epsilon):
        # X is a matrix (3,n)
        # S_{t+1} = S_{t} - T \beta I S / N T
        # I_{t+1} = I_{t} +\beta I S T / N - \gamma I
        # S_{t+1} = S_t + \gamma I_t T
        # R_0 = \beta / \gamma
        self.betaIndependent = [self.beta for i in range(len(datasets))]
        self.gammaIndependent = [self.gamma for i in range(len(datasets))]
        self.lambdas = [0 for i in range(len(datasets))]
        
        self.betaCentral = self.beta
        self.gammaCentral = self.gamma
        self.epsilon = epsilon
        if self.logging == True:
            self.logger = [{'lambdas': [], 'constraints': []} for i in range(len(datasets))] 

        for e in range(self.epochs):
            nablaBetaCentAccum, nablaGammaCentAccum = 0, 0
            for idx, thisDataset in enumerate(datasets):
                data, population = thisDataset
                # Compute the update
                self.population = population
                x_0 = data[:,0]
                # First get matrix for the country wise
                self.beta = self.betaIndependent[idx] 
                self.gamma = self.gammaIndependent[idx] 
                roll = self.getRolloutMatrix(x_0, data.shape[1])
                nablaBeta, nablaGamma = self.getGradient(roll, data)
                
                # Now get the matrix for the centralized
                self.beta = self.betaCentral
                self.gamma = self.gammaCentral
                rollCentral = self.getRolloutMatrix(x_0, data.shape[1])
                nablaBetaCent, nablaGammaCent = self.getGradient(roll, rollCentral)

                # Now compute the gradient
                self.betaIndependent[idx] = self.betaIndependent[idx] - self.eta_cent[idx] * (nablaBeta + self.lambdas[idx]*nablaBetaCent)
                self.gammaIndependent[idx] = self.gammaIndependent[idx] - self.eta_cent[idx] * (nablaGamma + self.lambdas[idx]*nablaGammaCent)
                
                # Now compute the gradient for the centralized
                nablaBetaCent, nablaGammaCent = self.getGradient(rollCentral, roll)
                nablaBetaCentAccum = nablaBetaCentAccum + nablaBetaCent
                nablaGammaCentAccum = nablaGammaCentAccum + nablaGammaCent
                if e%10 == 0:
                    self.lambdas[idx] = self.lambdas[idx] + self.eta_dual/population*(np.linalg.norm(roll-rollCentral)**2-self.epsilon**2)
                    self.lambdas[idx] = np.maximum(self.lambdas[idx],0)
                    # print(e,self.lambdas)
            # Update the centralized
            self.betaCentral = self.betaCentral - self.eta * nablaBetaCentAccum
            self.gammaCentral = self.gammaCentral - self.eta * nablaGammaCentAccum
            if self.logging == True and e % self.logg_every_e == 0 :
                constraints = self.evaluateConstraint(datasets)
                for idx, thisDataset in enumerate(datasets):
                    self.logger[idx]['lambdas'] += [self.lambdas[idx]]
                    self.logger[idx]['constraints'] += [constraints[idx]]

        pass

    def evaluate(self, X):
        x_0 = X[:,0]
        roll = self.getRolloutMatrix(x_0, X.shape[1])
        return np.linalg.norm(roll-X)
    
    def evaluateIndependent(self, X, population):
        x_0 = X[:,0]
        gamma = self.gamma
        beta = self.beta
        self.population = population

        errors = []
        for idx in range(len(self.gammaIndependent)):
            self.gamma = self.gammaIndependent[idx]
            self.beta = self.betaIndependent[idx]
            roll = self.getRolloutMatrix(x_0, X.shape[1])
            errors = errors + [np.linalg.norm(roll-X)]

        self.gamma = gamma
        self.beta = beta 
        return errors


        

# ToD0:
# Normalize the Infected and Susecptible 

# Plot beta and gamma 
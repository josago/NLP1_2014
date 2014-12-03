import numpy             as np
import numpy.random      as rd
import scipy.stats       as st
import scipy.special     as sp
import matplotlib.pyplot as plt

class CSLDA:
    def __init__(self, K, alpha_0, beta_0, eta_0, sigma_0):
        self.K = K
        
        self.alpha = alpha_0
        self.beta  = beta_0
        self.eta   = eta_0
        self.sigma = sigma_0

    def __gibbs_sampler(self, N_kw, N_dk, N_k, N_d, Z, S, W):
        for d in range(len(W)):
            for i in range(W[d].shape[0]):
                w     = W[d][i]
                k_old = Z[d][i]
                
                N_kw[k_old, w] -= 1
                N_dk[d, k_old] -= 1
                
                p = np.ones((self.K), dtype = float)

                for k_new in range(self.K):
                    select1        = np.zeros((self.K), dtype = int)
                    select1[k_new] = 1
                    
                    select2           = np.zeros((self.K, self.W), dtype = int)
                    select2[k_new, w] = 1
                    
                    p[k_new] *= np.prod(np.divide(np.prod(sp.gamma(N_kw + select2 + self.beta), axis = 1), sp.gamma(N_k + select1 + self.W * self.beta)))
                    
                    posterior_mean = float(np.divide(N_dk[d] + select1, np.repeat(np.reshape(N_d[d], (-1, 1)), self.K, axis = 1)).dot(np.reshape(self.eta, (-1, 1))))
                    
                    p[k_new] *= st.norm.pdf(S[d], posterior_mean, self.sigma)
                    p[k_new] *= np.prod(sp.gamma(N_dk[d, :] + self.alpha + select1))
                print p
                print np.sum(p)
                new_k   = rd.choice(self.K, p = p / np.sum(p))
                Z[d][i] = new_k
                
                N_kw[k_new, w] += 1
                N_dk[d, k_new] += 1

    def __estimate_eta(self, N_dk, N_d, S, iterations = 100, gamma = 0.5, epsilon = 0.001):
        select = 1 - np.eye(self.K, dtype = int)
        select = np.reshape(select, (self.K, 1, self.K))
        select = np.repeat(select, self.D, axis = 1)
        
        Z_norm = np.true_divide(N_dk, np.repeat(np.reshape(N_d, (-1, 1)), self.K, axis = 1)).T

        S = np.repeat(np.reshape(S, (1, self.D)), self.K, axis = 0)
        
        den = np.sum(np.power(Z_norm, 2), axis = 1) + epsilon
        
        for it in range(iterations):
            Z_eta = np.multiply(np.repeat(np.reshape(self.eta, (-1, 1)), self.D, axis = 1), Z_norm)
            Z_eta = np.reshape(Z_eta, (1, self.D, self.K))
            Z_eta = np.repeat(Z_eta, self.K, axis = 0)
            
            Z_eta_sum = np.sum(np.multiply(Z_eta, select), axis = 2)
            
            num = np.sum(np.multiply(Z_norm, S - Z_eta_sum), axis = 1)

            self.eta = (1 - gamma) * self.eta + gamma * np.divide(num, den)

    def train(self, S, W, num_burn_in = 10, num_skip = 1, num_samples = 10):
        self.D = S.shape[0]
        self.W = np.amax([np.amax(w) for w in W]) + 1
        
        phis = np.zeros((num_samples, self.K, self.W), dtype = float)
        etas = np.zeros((num_samples, self.K),         dtype = float)
        
        N_kw = np.zeros((self.K, self.W), dtype = int)
        N_dk = np.zeros((self.D, self.K), dtype = int)
        N_k  = np.zeros((self.K),         dtype = int)
        N_d  = np.zeros((self.D),         dtype = int)
        Z    = []
        
        # Randomly initialize Z:

        for d in range(self.D):
            N_d[d] = W[d].shape[0]
            
            z = np.zeros((N_d[d]), dtype = int)
            
            for i in range(N_d[d]):
                w = W[d][i]
                k = rd.choice(self.K)

                N_kw[k, w] += 1
                N_dk[d, k] += 1
                N_k[k]     += 1
                
                z[i] = k
                
            Z.append(z)

        if self.eta is None:
            self.eta = np.zeros((self.K), dtype = float)
            
            self.__estimate_eta(N_dk, N_d, S)
        
        sample = 0
        
        for iteration in range(num_burn_in + num_skip * num_samples):
            print "Iteration #%i..." % (iteration + 1)
            
            # Calculate sample:
            
            self.__gibbs_sampler(N_kw, N_dk, N_k, N_d, Z, S, W)
            self.__estimate_eta(N_dk, N_d, S)
            
            if iteration >= num_burn_in and (iteration - num_burn_in) % num_skip == 0:
                # Save sample:
                
                phis[sample] = np.divide(N_kw + self.beta, np.repeat(np.reshape(N_k, (-1, 1)), self.W, axis = 1) + self.W * self.beta)
                etas[sample] = np.copy(self.eta)
                
                sample += 1
        
        return (phis, etas)

S = np.array([2.0, 2.5, 3.5, 2.0, 2.5, 7.5, 7.5, 7.5, 8.0, 8.5])
W = [
    np.array([6, 4, 7, 2, 8, 8, 8, 1, 9, 0]),
    np.array([1, 2, 3, 9, 7, 1]),
    np.array([0, 1, 0, 1, 1, 1, 0, 1]),
    np.array([8, 6, 6, 4]),
    np.array([1, 1, 1, 1, 1, 2, 3, 4, 5, 4, 5, 6, 5, 4, 1, 7, 0, 0, 0, 8, 7]),
    np.array([10, 11, 12, 13, 14, 15, 16, 17]),
    np.array([18, 17, 19, 15, 15]),
    np.array([10, 10, 15, 18, 17, 12, 12, 12]),
    np.array([10, 13]),
    np.array([19, 18, 19, 13, 16, 17, 10, 10, 10, 18, 12, 13]),
    ]

cslda = CSLDA(K = 5, alpha_0 = 0.01, beta_0 = 0.01, eta_0 = None, sigma_0 = 0.5)
(phis, etas) = cslda.train(S, W)
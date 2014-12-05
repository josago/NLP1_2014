import numpy             as np
import numpy.random      as rd
import scipy.stats       as st
import scipy.special     as sp
import matplotlib.pyplot as plt

def load_database(filename):
    print "Loading database from '%s'..." % filename
    
    S        = []
    W        = 0
    Wsummary = []
    Wscript  = []
    
    f = open(filename, "r")
    
    for line in f:
        w_summary = []
        w_script  = []
        
        (name, score, summary, script) = line.split("\t")
        
        summary = summary.split(",")
        script  = script.split(",")
        
        W = max(W, max(len(summary) - 1, len(script) - 1))
        
        for w in range(len(summary)):
            try:
                for i in range(int(summary[w])):
                    w_summary.append(w)
                    
                for i in range(int(script[w])):
                    w_script.append(w)
            except ValueError:
                pass
        
        try:
            if len(w_summary) > 0 and len(w_script) > 0: # Checks that the movie has both summary and script.
                S.append(float(score))
                Wsummary.append(np.array(w_summary))
                Wscript.append(np.array(w_script))
        except ValueError:
            pass # The movie has no score, so we will not load it.
        
    f.close()
    
    return (np.array(S), W, Wsummary, Wscript)

class CSLDA:
    def __init__(self, use_scores, K, W, alpha_0 = 0.01, beta_0 = 0.01, eta_0 = None, sigma_0 = 0.5):
        self.use_scores = use_scores
        self.K          = K  # Number of topics.
        self.W          = W  # Number of unique words.
        
        self.alpha = alpha_0 # Initial value for the hyperparameter alpha (influences theta).
        self.beta  = beta_0  # Initial value for the hyperparameter beta  (influences phi).
        self.eta   = eta_0   # Initial value for the hyperparameter eta   (influences the means of scores). Set to None to estimate it. 
        self.sigma = sigma_0 # Initial value for the hyperparameter sigma (influences the variance of scores).
        
        # Training samples:
        
        self.N_skw = None
        self.N_sk  = None
        
    def __split_W(self, W, p1 = 0.5):
        W1 = []
        W2 = []
        
        for d in range(len(W)):
            w1 = []
            w2 = [] 
            
            for i in range(W[d].shape[0]):
                if rd.binomial(1, p1) == 0:
                    w1.append(W[d][i])
                else:
                    w2.append(W[d][i])
            
            W1.append(np.array(w1))
            W2.append(np.array(w2))
        
        return (W1, W2)
    
    def __init_all(self, W):
        N_kw = np.zeros((self.K, self.W), dtype = int)
        N_dk = np.zeros((self.D, self.K), dtype = int)
        N_k  = np.zeros((self.K),         dtype = int)
        N_d  = np.zeros((self.D),         dtype = int)
        Z    = []
        
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
            
        return (N_kw, N_dk, N_k, N_d, Z)
    
    def __gibbs_sampler(self, N_kw, N_dk, N_k, N_d, Z, S, W):
        for d in range(len(W)):
            for i in range(W[d].shape[0]):
                w     = W[d][i]
                k_old = Z[d][i]

                N_k[k_old]     -= 1
                N_kw[k_old, w] -= 1
                N_dk[d, k_old] -= 1
                
                p = np.zeros((self.K), dtype = float)

                for k in range(self.K):
                    select1    = np.zeros((self.K), dtype = int)
                    select1[k] = 1
                    
                    select2       = np.zeros((self.K, self.W), dtype = int)
                    select2[k, w] = 1
                    
                    # Log-space probabilities:
                    
                    p[k] += np.sum(np.subtract(np.sum(sp.gammaln(N_kw + select2 + self.beta), axis = 1), sp.gammaln(N_k + select1 + self.W * self.beta)))
                    
                    #p[k] *= np.prod(np.divide(np.prod(sp.gamma(N_kw + select2 + self.beta), axis = 1), sp.gamma(N_k + select1 + self.W * self.beta))) # Linear-space probabilities.
                    
                    if self.use_scores:
                        posterior_mean = float(np.divide(N_dk[d] + select1, np.repeat(np.reshape(N_d[d], (-1, 1)), self.K, axis = 1)).dot(np.reshape(self.eta, (-1, 1))))
                    
                        p[k] += np.log(st.norm.pdf(S[d], posterior_mean, self.sigma) + 1E-100) # This avoids infinities.
                        
                        #p[k] *= st.norm.pdf(S[d], posterior_mean, self.sigma)        # Linear-space probabilities.
                        
                    p[k] += np.sum(sp.gammaln(N_dk[d, :] + self.alpha + select1))

                    #p[k] *= np.prod(sp.gamma(N_dk[d, :] + self.alpha + select1)) # Linear-space probabilities.
                    
                p = np.exp(p - np.amax(p)) # Change to linear-space probabilities.

                k_new   = rd.choice(self.K, p = p / np.sum(p))
                Z[d][i] = k_new
                
                N_k[k_new]     += 1
                N_kw[k_new, w] += 1
                N_dk[d, k_new] += 1

    def __estimate_eta(self, N_dk, N_d, S, iterations = 1000, gamma = 0.0001, epsilon = 0.001):
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

            num = np.sum(np.multiply(Z_norm, np.subtract(S, Z_eta_sum)), axis = 1)

            self.eta = (1 - gamma) * self.eta + gamma * np.divide(num, den)
   
    def train(self, S, W, num_burn_in, num_skip, num_samples):
        self.D = S.shape[0]
        
        print "Training CSLDA model with %d topics, %d documents and %d unique words..." % (self.K, self.D, self.W)
        
        # Allocate data structures:
        
        self.N_skw = np.zeros((num_samples, self.K, self.W), dtype = float)
        self.N_sk  = np.zeros((num_samples, self.K),         dtype = float)
        
        (N_kw, N_dk, N_k, N_d, Z) = self.__init_all(W)

        if self.eta is None:
            self.eta = np.zeros((self.K), dtype = float)
            
            self.__estimate_eta(N_dk, N_d, S)
        
        sample = 0
        
        for iteration in range(num_burn_in + num_skip * num_samples):
            print "\tIteration #%i..." % (iteration + 1)
            
            # Calculate sample:
            
            self.__gibbs_sampler(N_kw, N_dk, N_k, N_d, Z, S, W)
            self.__estimate_eta(N_dk, N_d, S)
            
            if iteration >= num_burn_in and (iteration - num_burn_in) % num_skip == 0:
                # Save sample:
                
                self.N_skw[sample] = np.copy(N_kw)
                self.N_sk[sample]  = np.copy(N_k)
                
                sample += 1
                
    def test(self, S, W, num_burn_in, num_skip, num_samples):
        self.D = S.shape[0]
        
        print "Testing CSLDA model with %d topics, %d documents and %d unique words..." % (self.K, self.D, self.W)
        
        perplex = 0.0
        inv_acc = 0.0
        
        # Allocate data structures:

        (W1, W2) = self.__split_W(W)

        N2 = 0
        
        for sample in range(self.N_skw.shape[0]):
            (N_kw, N_dk, N_k, N_d, Z) = self.__init_all(W1) # Random initialization of Z.
            
            N_kw += self.N_skw[sample]
            N_k  += self.N_sk[sample]
        
            for iteration in range(num_burn_in + num_skip * num_samples):
                print "\tIteration #%i..." % (sample * (num_burn_in + num_skip * num_samples) + iteration + 1)
                
                # Calculate sample:
                
                self.__gibbs_sampler(N_kw, N_dk, N_k, N_d, Z, S, W1)

                if iteration >= num_burn_in and (iteration - num_burn_in) % num_skip == 0:
                    # Use sample:
                    
                    phis   = np.divide(N_kw + self.beta,  np.repeat(np.reshape(N_k, (-1, 1)), self.W, axis = 1) + self.W * self.beta)
                    thetas = np.divide(N_dk + self.alpha, np.repeat(np.reshape(N_d, (-1, 1)), self.K, axis = 1) + self.K * self.alpha)
                    
                    p = thetas.dot(phis)
                        
                    for d in range(len(W2)):
                        for i in range(W2[d].shape[0]):
                            w = W2[d][i]
                            
                            perplex += np.log2(p[d, w])
                            inv_acc += p[d, w]
                            
                            N2 += 1
              
        print "Model perplexity       = %.2f" % np.power(2, - perplex / N2)
        print "Model inverse accuracy = %.2f" % (N2 / inv_acc)

# For the time being we will just use movie summaries:

(S, W, Wsummary, Wscript) = load_database("database.csv")

Strain = S[0 : 5]
Stest  = S[5 : 10]

Wtrain = Wsummary[0 : 5]
Wtest  = Wsummary[5 : 10]

# Train and test the CSLDA model:

use_scores = True
K          = 20

num_burn_in = 5
num_skip    = 3
num_samples = 5

cslda = CSLDA(use_scores, K, W)
cslda.train(Strain, Wtrain, num_burn_in, num_skip, num_samples)
cslda.test(Stest, Wtest, num_burn_in, num_skip, num_samples)
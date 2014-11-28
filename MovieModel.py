# Execute the model on the GPU using: "optirun --no-xorg python MovieModel.py"

try:
    import cPickle as pickle
except:
    import pickle

from theano import function, shared

import theano.tensor     as T
import numpy             as np
import numpy.random      as rd
import matplotlib.pyplot as plt

def load_database(filename, prob_train = 0.8):
    Ilang_train = np.array([])
    Igen_train  = np.array([])
    S_train     = np.array([])
    
    Ilang_test  = np.array([])
    Igen_test   = np.array([])
    S_test      = np.array([])
    
    print "Reading movie database from '%s'..." % filename
        
    f = open(filename, "r")
    
    content = f.readlines()
        
    f.close()
    
    L = None # Number of language features.
    
    for line in content:
        (title, score, features) = line.rsplit("\t") # TODO: Update as the file format of the database changes.
        
        try:
            score    = np.reshape(float(score), (1))
            features = np.reshape(np.array(features.split(",")), (1, -1))
            
            if score >= 0 and score <= 10:
                if L is None or features.shape[1] == L: # Skips badly-shaped entries.
                    L = features.shape[1]
                    
                    if rd.uniform() < prob_train:
                        if Ilang_train.shape[0] == 0: # Empty matrix.
                            Ilang_train = features
                            S_train     = score
                        else:
                            Ilang_train = np.append(Ilang_train, features, axis = 0)
                            S_train     = np.append(S_train, score)
                    else:
                        if Ilang_test.shape[0] == 0: # Empty matrix.
                            Ilang_test = features
                            S_test     = score
                        else:
                            Ilang_test = np.append(Ilang_test, features, axis = 0)
                            S_test     = np.append(S_test, score)
        except ValueError: # The movie has no score.
            pass # Simply ignore this entry.
    
    Igen_train = np.zeros((Ilang_train.shape[0], 0), dtype = np.float32) # TODO: Temporal.
    
    Igen_test  = np.zeros((Ilang_test.shape[0], 0),  dtype = np.float32) # TODO: Temporal.
    
    S_train = np.reshape(S_train, (-1, 1))
    S_test  = np.reshape(S_test,  (-1, 1))
    
    return (Ilang_train, Igen_train, S_train, Ilang_test, Igen_test, S_test)

def model_from_file(filename):
    print "Reading movie model from '%s'..." % filename
        
    f = open(filename, "rb")
        
    model = pickle.load(f)
    
    f.close()
    
    return model

class MovieModel:
    def __init__(self, num_lang_inputs, num_gen_inputs, num_topics, SIGMA = 10E-9):
        self.num_lang_inputs = num_lang_inputs
        self.num_gen_inputs  = num_gen_inputs
        self.num_topics      = num_topics

        # Language features input layer:
        
        self.Ilang = T.fmatrix('Ilang') # Admits more than one movie at a time.
        
        # Topic layer:
        
        self.W1     = T.fmatrix('W1') # Weights for the topic layer.
        self.B1     = T.fmatrix('B1') # Biases for the topic layer.
        
        self.v_W1   = shared(np.float32(rd.normal(0.0, SIGMA, size = (num_lang_inputs, num_topics))))
        self.v_B1   = shared(np.zeros((1, num_topics), dtype = np.float32))
        
        self.topics = T.nnet.softmax(T.dot(self.Ilang, self.W1) + np.repeat(self.B1, self.Ilang.shape[0], axis = 0))
        
        # Autoencoder:
        
        self.Wauto = T.fmatrix('Wauto')
        self.Bauto = T.fmatrix('Bauto')
        
        self.v_Wauto = shared(np.float32(rd.normal(0.0, SIGMA, size = (num_topics, num_lang_inputs))))
        self.v_Bauto = shared(np.zeros((1, num_lang_inputs), dtype = np.float32))

        self.autoenc     = T.dot(self.topics, self.Wauto) + np.repeat(self.Bauto, self.Ilang.shape[0], axis = 0) # Linear autoencoder output.
        self.obj_autoenc = T.sum((self.Ilang - self.autoenc) ** 2) # Objective function for the autoencoder.
        
        # General features input layer:
        
        self.Igen = T.fmatrix('Igen')
        
        self.Wi   = T.fmatrix('Wi') # Weights for the general features.
        self.Wt   = T.fmatrix('Wt') # Weights for the LDA-like topics.
        self.B2   = T.fmatrix('B2') # Biases for the output.
        
        self.v_Wi = shared(np.float32(rd.normal(0.0, SIGMA, size = (num_gen_inputs, 1))))
        self.v_Wt = shared(np.float32(rd.normal(0.0, SIGMA, size = (num_topics, 1))))
        self.v_B2 = shared(np.zeros((1, 1), dtype = np.float32))
        
        # Output layer:
        
        self.S       = T.fmatrix('S') # Supervised movie score.
        
        self.out     = 10 * T.nnet.sigmoid(T.dot(self.topics, self.Wt) + T.dot(self.Igen, self.Wi) + np.repeat(self.B2, self.Ilang.shape[0], axis = 0))
        self.obj_out = T.sum((self.S - self.out) ** 2) # Objective function for the whole model.
        
        # Parameter gradients:
        
        self.g_W1    = T.grad(self.obj_autoenc, self.W1)
        self.g_B1    = T.grad(self.obj_autoenc, self.B1)
        self.g_Wauto = T.grad(self.obj_autoenc, self.Wauto)
        self.g_Bauto = T.grad(self.obj_autoenc, self.Bauto)
        
        self.g2_W1 = T.grad(self.obj_out, self.W1)
        self.g2_B1 = T.grad(self.obj_out, self.B1)
        self.g_Wi  = T.grad(self.obj_out, self.Wi)
        self.g_Wt  = T.grad(self.obj_out, self.Wt)
        self.g_B2  = T.grad(self.obj_out, self.B2)
        
    def pretrain(self, Ilang, eta = 10E-6, num_epochs = 30):
        N = Ilang.shape[0]
        
        print "Pre-training movie model with %i inputs for %i epochs..." % (N, num_epochs)
        
        params = [(self.W1, self.v_W1, self.g_W1), (self.B1, self.v_B1, self.g_B1), (self.Wauto, self.v_Wauto, self.g_Wauto), (self.Bauto, self.v_Bauto, self.g_Bauto)]
        
        for epoch in range(num_epochs):
            #print "\tEpoch %i..." % int(epoch + 1)
            
            for (param, v_param, g_param) in params:
                f = function([self.W1, self.B1, self.Wauto, self.Bauto, self.Ilang], g_param, allow_input_downcast = True)
                
                v_param.set_value(v_param.get_value() - eta * f(self.v_W1.get_value(), self.v_B1.get_value(), self.v_Wauto.get_value(), self.v_Bauto.get_value(), Ilang))
        
    def train(self, Ilang_train, Igen_train, S_train, Ilang_test, Igen_test, S_test, eta = 10E-6, num_epochs = 30):
        N = Ilang_train.shape[0]
        
        print "Training movie model with %i inputs for %i epochs..." % (N, num_epochs)
        
        params = [(self.Wi, self.v_Wi, self.g_Wi), (self.Wt, self.v_Wt, self.g_Wt), (self.B2, self.v_B2, self.g_B2), (self.W1, self.v_W1, self.g2_W1), (self.B1, self.v_B1, self.g2_B1)]
        
        error_train = []
        error_test  = []
        
        for epoch in range(num_epochs):
            #print "\tEpoch %i..." % int(epoch + 1)
            
            for (param, v_param, g_param) in params:
                f = function([self.W1, self.B1, self.Wi, self.Wt, self.B2, self.Ilang, self.Igen, self.S], g_param, allow_input_downcast = True)
                
                v_param.set_value(v_param.get_value() - eta * f(self.v_W1.get_value(), self.v_B1.get_value(), self.v_Wi.get_value(), self.v_Wt.get_value(), self.v_B2.get_value(), Ilang_train, Igen_train, S_train))

            error_train.append(self.error(Ilang_train, Igen_train, S_train))
            error_test.append(self.error(Ilang_test, Igen_test, S_test))
            
        return (error_train, error_test)
    
    def evaluate(self, Ilang, Igen):
        f_topics = function([self.W1, self.B1, self.Ilang], self.topics, allow_input_downcast = True)
        f_out    = function([self.W1, self.B1, self.Wi, self.Wt, self.B2, self.Ilang, self.Igen], self.out, allow_input_downcast = True)
        
        topics = f_topics(self.v_W1.get_value(), self.v_B1.get_value(), Ilang)
        scores = f_out(self.v_W1.get_value(), self.v_B1.get_value(), self.v_Wi.get_value(), self.v_Wt.get_value(), self.v_B2.get_value(), Ilang, Igen)
        
        return (scores, topics)
    
    def error(self, Ilang, Igen, S):
        (scores, _) = self.evaluate(Ilang, Igen)

        return np.mean(np.absolute(scores - S))
    
    def dump2file(self, filename):
        print "Dumping movie model into '%s'..." % filename
        
        f = open(filename, "wb")
        
        pickle.dump(self, f)
        
        f.close()
        
# Loading the database:

(Ilang_train, Igen_train, S_train, Ilang_test, Igen_test, S_test) = load_database("database.csv")

num_lang_inputs = Ilang_train.shape[1]
num_gen_inputs  = Igen_train.shape[1]

print "Training with %i samples and testing with %i samples..." % (S_train.shape[0], S_test.shape[0])

# Show a histogram of movie scores:

#plt.hist(np.append(S_train, S_test, axis = 0).flatten(), 40)
#plt.xticks(range(0, 11))
#plt.show()

NUM_TRIALS = 5

for num_topics in (32, 64, 128, 256, 512, 1024, 2048, 4096):
    ERROR_TEST = None
    
    for it in range(NUM_TRIALS):
        # Initializing the model:

        mm = MovieModel(num_lang_inputs, num_gen_inputs, num_topics)

        # Training and evaluating the model:

        mm.pretrain(Ilang_train)       
        (error_train, error_test) = mm.train(Ilang_train, Igen_train, S_train, Ilang_test, Igen_test, S_test)

        if ERROR_TEST is None:
            ERROR_TEST  = np.array(error_test)
        else:
            ERROR_TEST += np.array(error_test)

        # Saving and loading the model:

        mm.dump2file("MovieModel_%i.pkl" % num_topics)
        #mm = model_from_file("MovieModel.pkl")

    # Plotting the results:
    
    print ERROR_TEST / NUM_TRIALS

    plt.hold(True)
    plt.plot(ERROR_TEST / NUM_TRIALS, label = "Average absolute error (%i topics)" % num_topics)
    plt.legend()

plt.show()
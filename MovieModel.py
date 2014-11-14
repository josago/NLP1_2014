# Execute the model on the GPU using: "optirun --no-xorg python MovieModel.py"

try:
    import cPickle as pickle
except:
    import pickle

from theano import function, shared

import theano.tensor as T
import numpy         as np
import numpy.random  as rd

def model_from_file(filename):
    print "Reading movie model from '%s'..." % filename
        
    f = open(filename, "rb")
        
    model = pickle.load(f)
    
    f.close()
    
    return model

class MovieModel:
    def __init__(self, num_lang_inputs, num_gen_inputs, num_topics, SIGMA = 10E-2):
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
        
        self.Wi   = T.fmatrix('Wi') # Weights for the LDA-like topics.
        self.Wt   = T.fmatrix('Wt') # Weights for the general features.
        self.B2   = T.fmatrix('B2') # Biases for the output.
        
        self.v_Wi = shared(np.float32(rd.normal(0.0, SIGMA, size = (num_gen_inputs, 1))))
        self.v_Wt = shared(np.float32(rd.normal(0.0, SIGMA, size = (num_topics, 1))))
        self.v_B2 = shared(np.zeros((1, 1), dtype = np.float32))
        
        # Output layer:
        
        self.S       = T.fmatrix('S') # Supervised movie score.
        
        self.out     = T.nnet.sigmoid(T.dot(self.topics, self.Wt) + T.dot(self.Igen, self.Wi) + np.repeat(self.B2, self.Ilang.shape[0], axis = 0)) * 10
        self.obj_out = T.sum((self.S - self.out) ** 2) # Objective function for the whole model.
        
        # Parameter gradients:
        
        self.g_W1    = T.grad(self.obj_autoenc, self.W1)
        self.g_B1    = T.grad(self.obj_autoenc, self.B1)
        self.g_Wauto = T.grad(self.obj_autoenc, self.Wauto)
        self.g_Bauto = T.grad(self.obj_autoenc, self.Bauto)
        
        self.g_Wi = T.grad(self.obj_out, self.Wi)
        self.g_Wt = T.grad(self.obj_out, self.Wt)
        self.g_B2 = T.grad(self.obj_out, self.B2)
        
    def pretrain(self, Ilang, eta = 10E-6, num_epochs = 1000):
        N = Ilang.shape[0]
        
        print "Pre-training movie model with %i inputs for %i epochs..." % (N, num_epochs)
        
        params = [(self.W1, self.v_W1, self.g_W1), (self.B1, self.v_B1, self.g_B1), (self.Wauto, self.v_Wauto, self.g_Wauto), (self.Bauto, self.v_Bauto, self.g_Bauto)]
        
        for epoch in range(num_epochs):
            print "\tEpoch %i..." % int(epoch + 1)
            
            for (param, v_param, g_param) in params:
                f = function([self.W1, self.B1, self.Wauto, self.Bauto, self.Ilang], g_param, allow_input_downcast = True)
                
                v_param.set_value(v_param.get_value() - eta * f(self.v_W1.get_value(), self.v_B1.get_value(), self.v_Wauto.get_value(), self.v_Bauto.get_value(), Ilang))
        
    def train(self, Ilang, Igen, S, eta = 10E-6, num_epochs = 1000):
        N = Ilang.shape[0]
        
        print "Training movie model with %i inputs for %i epochs..." % (N, num_epochs)
        
        params = [(self.Wi, self.v_Wi, self.g_Wi), (self.Wt, self.v_Wt, self.g_Wt), (self.B2, self.v_B2, self.g_B2)]
        
        for epoch in range(num_epochs):
            print "\tEpoch %i..." % int(epoch + 1)
            
            for (param, v_param, g_param) in params:
                f = function([self.W1, self.B1, self.Wi, self.Wt, self.B2, self.Ilang, self.Igen, self.S], g_param, allow_input_downcast = True)
                
                v_param.set_value(v_param.get_value() - eta * f(self.v_W1.get_value(), self.v_B1.get_value(), self.v_Wi.get_value(), self.v_Wt.get_value(), self.v_B2.get_value(), Ilang, Igen, S))
    
    def evaluate(self, Ilang, Igen):
        f_topics = function([self.W1, self.B1, self.Ilang], self.topics, allow_input_downcast = True)
        f_out    = function([self.W1, self.B1, self.Wi, self.Wt, self.B2, self.Ilang, self.Igen], self.out, allow_input_downcast = True)
        
        topics = f_topics(self.v_W1.get_value(), self.v_B1.get_value(), Ilang)
        scores = f_out(self.v_W1.get_value(), self.v_B1.get_value(), self.v_Wi.get_value(), self.v_Wt.get_value(), self.v_B2.get_value(), Ilang, Igen)
        
        return (scores, topics)
    
    def dump2file(self, filename):
        print "Dumping movie model into '%s'..." % filename
        
        f = open(filename, "wb")
        
        pickle.dump(self, f)
        
        f.close()

# Initializing the model:

mm = MovieModel(8000, 10, 50) # MovieModel(num_lang_inputs, num_gen_inputs, num_topics)

# Training the model:

Ilang = np.zeros((27, 8000), dtype = np.int16)      # TODO: Use real language features!
Igen  = np.zeros((27, 10),   dtype = np.float32)    # TODO: Use real general movie features!
S     = np.float32(rd.uniform(size = (27, 1))) * 10 # TODO: Use real movie scores!

mm.pretrain(Ilang)       
mm.train(Ilang, Igen, S)

# Saving and loading the model:

mm.dump2file("MovieModel.pkl")
mm = model_from_file("MovieModel.pkl")

# Evaluating the model:

(scores, topics) = mm.evaluate(Ilang, Igen)

print scores
print topics
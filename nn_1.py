import sys
import os
import numpy as np
import pandas as pd

# The seed will be fixed to 42 for this assigmnet.
np.random.seed(42)

NUM_FEATS = 90

def relu(h):
    y=h.copy()
    for i in range(len(y)):
        for j in range(len(y[i])):
            if y[i][j]<0:
                y[i][j]=0
    return y
    
def der_relu(h):
    y=h.copy()
    for i in range(len(y)):
        for j in range(len(y[i])):
            if y[i][j]<=0:
                y[i][j]=0
            else:
                y[i][j]=1
    return y
    
    
class Net(object):
    '''
    '''

    def __init__(self, num_layers, num_units):
        '''
        Initialize the neural network.
        Create weights and biases.
        Here, we have provided an example structure for the weights and biases.
        It is a list of weight and bias matrices, in which, the
        dimensions of weights and biases are (assuming 1 input layer, 2 hidden layers, and 1 output layer):
        weights: [(NUM_FEATS, num_units), (num_units, num_units), (num_units, num_units), (num_units, 1)]
        biases: [(num_units, 1), (num_units, 1), (num_units, 1), (num_units, 1)]
        Please note that this is just an example.
        You are free to modify or entirely ignore this initialization as per your need.
        Also you can add more state-tracking variables that might be useful to compute
        the gradients efficiently.
        Parameters
        ----------
            num_layers : Number of HIDDEN layers.
            num_units : Number of units in each Hidden layer.
        '''
        self.num_layers = num_layers
        self.num_units = num_units

        self.biases = []
        self.weights = []
        for i in range(num_layers):
            if i==0:
                # Input layer
                self.weights.append(np.random.uniform(-1, 1, size=(NUM_FEATS, self.num_units)))
            else:
                # Hidden layer
                self.weights.append(np.random.uniform(-1, 1, size=(self.num_units, self.num_units)))

            self.biases.append(np.random.uniform(-1, 1, size=(self.num_units, 1)))

        # Output layer
        self.biases.append(np.random.uniform(-1, 1, size=(1, 1)))
        self.weights.append(np.random.uniform(-1, 1, size=(self.num_units, 1)))
        

    def __call__(self, X):
        '''Forward propagate the input X through the network,
        and return the output.
        Note that for a classification task, the output layer should
        be a softmax layer. So perform the computations accordingly
        Parameters
        ----------
            X : Input to the network, numpy array of shape m x d
        Returns
        ----------
            y : Output of the network, numpy array of shape m x 1
        '''
        #raise NotImplementedError
        self.forward_val=[]
        for i in range(self.num_layers):
            h=np.dot(X,self.weights[i])+(self.biases[i]).T
            h=relu(h)
            X=h
            self.forward_val.append(h)
        h=np.dot(X,self.weights[self.num_layers])+(self.biases[self.num_layers]).T
        self.forward_val.append(h)
        return h

    def backward(self, X, y, lamda):
        '''
        Compute and return gradients loss with respect to weights and biases.
        (dL/dW and dL/db)
        Parameters
        ----------
            X : Input to the network, numpy array of shape m x d
            y : Output of the network, numpy array of shape m x 1
            lamda : Regularization parameter.
        Returns
        ----------
            del_W : derivative of loss w.r.t. all weight values (a list of matrices).
            del_b : derivative of loss w.r.t. all bias values (a list of vectors).
        Hint: You need to do a forward pass before performing backward pass.
        '''
        #raise NotImplementedError
        del_W=[]
        del_b=[]
        o=self(X)
        for i in range(self.num_layers,-1,-1):
            if i==self.num_layers:
                delEwrtO=o.copy()
                for j in range(len(delEwrtO)):
                    delEwrtO[j]=2*(delEwrtO[j][0]-y[j])
                delOwrtNET=der_relu(o)
                
                delNETwrtoW=self.forward_val[i-1]
                delEwrtNET=delEwrtO*delOwrtNET
                del_b.append(np.reshape(delEwrtNET.mean(axis=0),(delEwrtNET.mean(axis=0).size,1)))
                
                dw=delEwrtNET*delNETwrtoW
                dw=np.array(dw)
                dw_avg=dw.mean(axis=0)
                del_W.append(np.reshape(dw_avg,(dw_avg.size,1)))
    
                """print("delEwrtO=",delEwrtO)
                print("delOwrtNET",delOwrtNET)
                print("delNETwrtoW=",delNETwrtoW)"""
                
                
                
            else:
                next_layer=del_b[self.num_layers-i-1]
                delEwrtO=[]
                
                for neuron in range(len(next_layer)):
                    if neuron==0:
                        sum=(next_layer[neuron])*self.weights[i+1]
                    else:
                        sum+=(next_layer[neuron])*self.weights[i+1]
                delEwrtO.append(sum)
                delEwrtO=np.array(delEwrtO)
                
                flatdelEwrtO=delEwrtO.flatten()
                delEwrtO=np.reshape(flatdelEwrtO,delEwrtO.shape[0:2])
                
                delOwrtNET=der_relu(self.forward_val[i])
                
                delEwrtNET=delEwrtO*delOwrtNET
                
                del_b.append(np.reshape(delEwrtNET.mean(axis=0),(delEwrtNET.mean(axis=0).size,1)))
                
                if i==0:
                    delNETwrtoW=X
                else:
                    delNETwrtoW=self.forward_val[i-1]
                    
                
                dw=[]
                for d in range(delNETwrtoW.shape[1]):
                    dw.append(np.array([delNETwrtoW[:,d]]).T*delEwrtNET)
                np.array(dw)    
                dw=np.array(dw)
                dw_new=[]
                for i in dw:
                    dw_new.append(i.mean(axis=0))
                
                del_W.append(np.array(dw_new))
                
        del_W=np.array(del_W)
        del_b=np.array(del_b)
        
        return del_W[::-1],del_b[::-1]
                


class Optimizer(object):
    '''
    '''

    def __init__(self, learning_rate):
        '''
		Create a Gradient Descent based optimizer with given
		learning rate.
		Other parameters can also be passed to create different types of
		optimizers.
		Hint: You can use the class members to track various states of the
		optimizer.
        '''
        #raise NotImplementedError
        self.learning_rate=learning_rate
        self.beta=0.9
        self.weight_velocity=0
        self.bias_velocity=0
        

    def step(self, weights, biases, delta_weights, delta_biases):
        '''
		Parameters
		----------
			weights: Current weights of the network.
			biases: Current biases of the network.
			delta_weights: Gradients of weights with respect to loss.
			delta_biases: Gradients of biases with respect to loss.
        '''
        #raise NotImplementedError
        self.weight_velocity=self.beta*(self.weight_velocity)+(1-self.beta)*np.array(delta_weights)
        weights_updated = np.array(weights)-np.array(self.learning_rate)*self.weight_velocity
        
        self.bias_velocity=self.beta*(self.bias_velocity)+(1-self.beta)*np.array(delta_biases)
        biases_updated = np.array(biases)-np.array(self.learning_rate)*self.bias_velocity
        return weights_updated, biases_updated
        


def loss_mse(y, y_hat):
    '''
	Compute Mean Squared Error (MSE) loss betwee ground-truth and predicted values.
	Parameters
	----------
		y : targets, numpy array of shape m x 1
		y_hat : predictions, numpy array of shape m x 1
	Returns
	----------
		MSE loss between y and y_hat.
    '''
    #raise NotImplementedError
    loss=0
    for i in range(len(y)):
        loss+=(y[i]-y_hat[i][0])**2
    return loss

def loss_regularization(weights, biases):
    '''
	Compute l2 regularization loss.
	Parameters
	----------
		weights and biases of the network.
	Returns
	----------
		l2 regularization loss 
    '''
	#raise NotImplementedError
    rloss=0
    for val in weights*weights:
        rloss=rloss+np.sum(val)
    #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\nrloss",rloss)
    for val in biases*biases:
        rloss+=rloss+np.sum(val)
    #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\nrloss",rloss)
    return rloss

def loss_fn(y, y_hat, weights, biases, lamda):
	
    '''
	Compute loss =  loss_mse(..) + lamda * loss_regularization(..)
	Parameters
	----------
		y : targets, numpy array of shape m x 1
		y_hat : predictions, numpy array of shape m x 1
		weights and biases of the network
		lamda: Regularization parameter
	Returns
	----------
		l2 regularization loss 
    '''
	#raise NotImplementedError
    loss =  loss_mse(y, y_hat) + lamda * loss_regularization(weights, biases)
    return loss/len(y)

def rmse(y, y_hat):

    '''
	Compute Root Mean Squared Error (RMSE) loss betwee ground-truth and predicted values.
	Parameters
	----------
		y : targets, numpy array of shape m x 1
		y_hat : predictions, numpy array of shape m x 1
	Returns
	----------
		RMSE between y and y_hat.
    '''
	#raise NotImplementedError
    loss=loss_mse(y, y_hat)**0.5
    return loss/len(y)
    

def cross_entropy_loss(y, y_hat):
    '''
	Compute cross entropy loss
	Parameters
	----------
		y : targets, numpy array of shape m x 1
		y_hat : predictions, numpy array of shape m x 1
	Returns
	----------
		cross entropy loss
    '''
	#raise NotImplementedError
    h=0
    for i in range(len(y)):
        h+=y*math.log(y_hat)
    h=-1*h
    return h


def train(net, optimizer, lamda, batch_size, max_epochs,train_input, train_target,dev_input, dev_target):
    '''
	In this function, you will perform following steps:
		1. Run gradient descent algorithm for `max_epochs` epochs.
		2. For each bach of the training data
			1.1 Compute gradients
			1.2 Update weights and biases using step() of optimizer.
		3. Compute RMSE on dev data after running `max_epochs` epochs.
	Here we have added the code to loop over batches and perform backward pass
	for each batch in the loop.
	For this code also, you are free to heavily modify it.
    '''

    m = train_input.shape[0]

    for e in range(max_epochs):
        epoch_loss = 0.
        for i in range(0, m, batch_size):
            batch_input = train_input[i:i+batch_size]
            batch_target = train_target[i:i+batch_size]
            pred = net(batch_input)
    
            # Compute gradients of loss w.r.t. weights and biases
            dW, db = net.backward(batch_input, batch_target, lamda)
            #print("--------------------------\n",dW[1])
            # Get updated weights based on current weights and gradients
            weights_updated, biases_updated = optimizer.step(net.weights, net.biases, dW, db)
            
            #print("weights_updated==",weights_updated[1])
            
            #print("weights_updated------",weights_updated)
            #print("biases_updated------",biases_updated)

            # Update model's weights and biases
            net.weights = weights_updated
            net.biases = biases_updated

            # Compute loss for the batch
            batch_loss = loss_fn(batch_target, pred, net.weights, net.biases, lamda/(e+1))
            #print(batch_loss)
            epoch_loss += (batch_loss/batch_size)

            #print(e, i, rmse(batch_target, pred), batch_loss)

        print("===============================")
        print(e, epoch_loss)

        # Write any early stopping conditions required (only for Part 2)
        # Hint: You can also compute dev_rmse here and use it in the early
        # 		stopping condition.

    # After running `max_epochs` (for Part 1) epochs OR early stopping (for Part 2), compute the RMSE on dev data.
    dev_pred = net(dev_input)
    dev_rmse = rmse(dev_target, dev_pred)
    
    np.set_printoptions(formatter={'float': '{: 0.5f}'.format})
    print('RMSE on dev data: ',dev_rmse)


def get_test_data_predictions(net, inputs):
    '''
	Perform forward pass on test data and get the final predictions that can
	be submitted on Kaggle.
	Write the final predictions to the part2.csv file.
	Parameters
	----------
		net : trained neural network
		inputs : test input, numpy array of shape m x d
	Returns
	----------
		predictions (optional): Predictions obtained from forward pass
								on test data, numpy array of shape m x 1
    '''
    #raise NotImplementedError
    y=net(inputs)
    y=np.array(y)
    a=np.arange(1,y.size+1)
    y=np.round_(y)
    DF = pd.DataFrame({"Id" : a, "Predictions" : y.flatten()})
    DF.to_csv("22m0788.csv",index=False)
    

def read_data():
    '''
    Read the train, dev, and test datasets
    '''
    #raise NotImplementedError
    
    train=pd.read_csv('cs725-2022-assignment-regression/train.csv')
    coltrain=list(train.columns)
    train_target=np.array(list(train['1']))
    train_input=[]
    for c in train[coltrain[1:]]:
        train_input.append(list(train[c]))
                   
    train_input=(np.array(train_input).T)
    
    
    
    
    dev=pd.read_csv('cs725-2022-assignment-regression/dev.csv')
    coldev=list(dev.columns)
    dev_target=np.array(list(dev['1']))
    
    dev_input=[]
    for c in dev[coldev[1:]]:
        dev_input.append(list(dev[c]))
                   
    dev_input=(np.array(dev_input).T)
    
    
    
    
    test=pd.read_csv('cs725-2022-assignment-regression/test.csv')
    coltest=list(test.columns)
    
    test_input=[]
    for c in test[coltest]:
        test_input.append(list(test[c]))
                   
    test_input=(np.array(test_input).T)
    
    
    
    
    return train_input, train_target, dev_input, dev_target, test_input


def main():

	# Hyper-parameters 
	max_epochs = 100
	batch_size = 256
	learning_rate = 0.000000001
	num_layers = 1
	num_units = 64
	lamda = 0.1 # Regularization Parameter

	train_input, train_target, dev_input, dev_target, test_input = read_data()
	net = Net(num_layers, num_units)
	optimizer = Optimizer(learning_rate)
	train(
		net, optimizer, lamda, batch_size, max_epochs,
		train_input, train_target,
		dev_input, dev_target
	)
	get_test_data_predictions(net, test_input)


if __name__ == '__main__':
	main()

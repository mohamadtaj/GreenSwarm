import numpy as np
import math
import os


"""
determine the number of samples for each node 
based on the total number of samples and the given node fractions 
"""
def split(n, fractions):
    
    result = []
    for fraction in fractions[:-1]:
        result.append(round(fraction * n))
    result.append(n - np.sum(result))
    
    return result

"""creating mini-batches for training"""   
def mini_batches(X, Y, mini_batch_size):
    
    m = X.shape[0]
    mini_batches = []

    permutation = np.random.permutation(m)
    shuffled_X = X[permutation,:]
    shuffled_Y = Y[permutation]

    num_complete_minibatches = math.floor(m/mini_batch_size)
    for k in range(0, num_complete_minibatches):

        mini_batch_X = shuffled_X[k*mini_batch_size : (k+1)*mini_batch_size]
        mini_batch_Y = shuffled_Y[k*mini_batch_size : (k+1)*mini_batch_size]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    if m % mini_batch_size != 0:

        mini_batch_X = shuffled_X[(k+1)*mini_batch_size : m]
        mini_batch_Y = shuffled_Y[(k+1)*mini_batch_size : m]
        
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches    
    
    
def scale (X):
  
    X = X.astype("float32")/255.

    return X   

"""function to reduce the size of a dataset based on a given train and test size"""    
def reduce_size (x_train, y_train, x_test, y_test, train_size, test_size):

    m = x_train.shape[0]
    n = x_test.shape[0]
    print("x train shape",m)
    print("x test shape",n)

    np.random.seed(0)
    permutation_train = np.random.permutation(m)
    np.random.seed(0)
    permutation_test = np.random.permutation(n)

    shuffled_X_train = x_train[permutation_train,:]
    shuffled_Y_train = y_train[permutation_train]
    shuffled_X_test = x_test[permutation_test,:]
    shuffled_Y_test = y_test[permutation_test]

    x_train_reduced = shuffled_X_train[:train_size]
    y_train_reduced = shuffled_Y_train[:train_size]
    x_test_reduced = shuffled_X_test[:test_size]
    y_test_reduced = shuffled_Y_test[:test_size]

    return x_train_reduced, y_train_reduced, x_test_reduced, y_test_reduced    

"""exporting the accuracy and loss"""    
def export_results(dataset_name, mode, train_accuracy, test_accuracy, train_loss, test_loss):

    if not os.path.exists("results"):
        os.makedirs("results")

    np.save("./results/"+dataset_name+"_"+mode+"_train_acc.npy", train_accuracy)
    np.save("./results/"+dataset_name+"_"+mode+"_test_acc.npy", test_accuracy)
    np.save("./results/"+dataset_name+"_"+mode+"_train_loss.npy", train_loss) 
    np.save("./results/"+dataset_name+"_"+mode+"_test_loss.npy", test_loss)    
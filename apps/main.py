# GPU Configuration
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
# config = ConfigProto()
# config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.5
# session = InteractiveSession(config=config)

from gs_peer import *
from fedavg_peer import *
from fedavg import *
from gs import *
from centralized import *
from data_loader import *
from tensorflow import keras



def main_gs (mode, data, data_type, num_nodes, node_fractions, EPOCHS, BATCH_SIZE, optimizer, loss):
    
    DATASET_NAME = data
    x_train, y_train, x_test, y_test = load_dataset(DATASET_NAME)
   
    unique_classes = len(np.unique(y_train)) # number of classes for classification

    # determining the number of layers at the output based on the number of classes
    if(unique_classes == 2):
        output_size = 1
    elif(unique_classes > 2):
        output_size = unique_classes 
        
    if(data_type == 'img'):
        input_size = x_train.shape[1:]
            
    elif(data_type == 'text'):
        input_size = None
            

    env = GS_Env(x_train, y_train, num_nodes, node_fractions) # create a new Greenswarm environment
    env.create_nodes()
    
    env.initialize_peers()    
    env.connect_net()        
    env.set_neighbors()
           
    print('--------------------------------------------')
    print(f'DATASET_NAME: {DATASET_NAME}')
    print(f'x_train shape: {x_train.shape}')
    print(f'y_train shape: {y_train.shape}')
   
    print()
    print()
    for node in env.nodes:
        print(f'node {node.id}:')
        print(f'x_train shape: {node.x_train.shape}')
        print(f'y_train shape: {node.y_train.shape}')
        print(f'neighbor id: {node.neighbor.id}')
        print()

    print('--------------------------------------------')

    train_accuracy, test_accuracy, train_loss, test_loss = train_network_gs (env, DATASET_NAME, x_test, y_test, BATCH_SIZE, EPOCHS, input_size, output_size, optimizer, loss)  

    export_results (DATASET_NAME, mode, train_accuracy, test_accuracy, train_loss, test_loss) # save the loss and accuracy as 'npy' files on the disk
    
    
def main_fedavg (mode, data, data_type, num_nodes, node_fractions, EPOCHS, BATCH_SIZE, optimizer, loss):
    
    DATASET_NAME = data
    x_train, y_train, x_test, y_test = load_dataset(DATASET_NAME)
    
    unique_classes = len(np.unique(y_train))

    if(unique_classes == 2):
        output_size = 1
    elif(unique_classes > 2):
        output_size = unique_classes 
        
    if(data_type == 'img'):
        input_size = x_train.shape[1:]
            
    elif(data_type == 'text'):
        input_size = None
            
    env = FL_Env(x_train, y_train, num_nodes, node_fractions)
    env.create_nodes()
    
    env.initialize_peers()    
    env.connect_net()    
    env.set_coordinator()
    
    print('--------------------------------------------')
    print(f'DATASET_NAME: {DATASET_NAME}')
    print(f'x_train shape: {x_train.shape}')
    print(f'y_train shape: {y_train.shape}')
   
    print()
    print()
    for node in env.nodes:
        print(f'node {node.id}:')
        print(f'x_train shape: {node.x_train.shape}')
        print(f'y_train shape: {node.y_train.shape}')
        print()
        
    print('--------------------------------------------')
    
    train_accuracy, test_accuracy, train_loss, test_loss = train_network_fedavg (env, DATASET_NAME, x_test, y_test, BATCH_SIZE, EPOCHS, input_size, output_size, optimizer, loss)  

    export_results(DATASET_NAME, mode, train_accuracy, test_accuracy, train_loss, test_loss)    


def main_centralized (mode, data, data_type, EPOCHS, BATCH_SIZE, optimizer, loss):
    
    DATASET_NAME = data
    x_train, y_train, x_test, y_test = load_dataset(DATASET_NAME)
   
    unique_classes = len(np.unique(y_train))

    if(unique_classes == 2):
        output_size = 1
    elif(unique_classes > 2):
        output_size = unique_classes 
        
    if(data_type == 'img'):
        input_size = x_train.shape[1:]
            
    elif(data_type == 'text'):
        input_size = None
         
    print('--------------------------------------------')
    print(f'DATASET_NAME: {DATASET_NAME}')
    print(f'x_train shape: {x_train.shape}')
    print(f'y_train shape: {y_train.shape}')
    print()
    print('--------------------------------------------')
    
    train_accuracy, test_accuracy, train_loss, test_loss = train_centralized (DATASET_NAME, x_train, y_train, x_test, y_test, BATCH_SIZE, EPOCHS, input_size, output_size, optimizer, loss)  

    export_results (DATASET_NAME, mode, train_accuracy, test_accuracy, train_loss, test_loss)    
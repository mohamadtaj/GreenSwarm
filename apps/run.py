import argparse
from main import *

parser = argparse.ArgumentParser()

# Parse command line arguments
parser.add_argument("--dataset", "--dataset", type=str ,help="Name of the dataset", default="cifar10")
parser.add_argument("--mode", "--mode", type=str, help="mode of training (GS framework: gs, FedAvg: fedavg, Centralized: cent)", default="gs")
args = vars(parser.parse_args())
 
# Set up parameters
data = args["dataset"]
mode = args["mode"]
num_nodes = 3
node_fractions = [0.1, 0.3, 0.6] 

# Neural networks parameters
BATCH_SIZE = 128

if (data == 'cifar10'):
    EPOCHS = 100
    data_type = 'img'
    optimizer = tf.keras.optimizers.SGD(learning_rate = 0.005, momentum = 0.9)
    loss = 'sparse_categorical_crossentropy'
    
elif (data == 'fashion'):
    EPOCHS = 100
    data_type = 'img'
    optimizer = tf.keras.optimizers.SGD(learning_rate = 0.001, momentum = 0.9)
    loss = 'sparse_categorical_crossentropy'

elif (data == 'malaria'):
    EPOCHS = 100
    data_type = 'img'
    optimizer = tf.keras.optimizers.SGD(learning_rate = 0.001, momentum = 0.9)
    loss = 'binary_crossentropy'

elif (data == 'retinal'):
    EPOCHS = 80
    data_type = 'img'
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)
    loss = 'sparse_categorical_crossentropy'

elif (data == 'xray'):
    EPOCHS = 200
    data_type = 'img'
    optimizer = tf.keras.optimizers.SGD(learning_rate = 0.001, momentum = 0.9)
    loss = 'binary_crossentropy'

elif (data == 'intel'):
    EPOCHS = 300
    data_type = 'img'
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)
    loss = 'sparse_categorical_crossentropy'

elif (data == 'imdb'):
    EPOCHS = 30
    data_type = 'text'
    optimizer = tf.keras.optimizers.Adam()
    loss = 'binary_crossentropy'

elif (data == 'reuters'):
    EPOCHS = 100
    data_type = 'text'
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
    loss = 'sparse_categorical_crossentropy'

print(f'Starting the simulation...')
 

if (mode == 'gs'):
    main_gs (mode, data, data_type, num_nodes, node_fractions, EPOCHS, BATCH_SIZE, optimizer, loss)

elif (mode == 'fedavg'):
    main_fedavg (mode, data, data_type, num_nodes, node_fractions, EPOCHS, BATCH_SIZE, optimizer, loss)

elif (mode == 'cent'):
    main_centralized (mode, data, data_type, EPOCHS, BATCH_SIZE, optimizer, loss)    

import numpy as np
from utils import *
from models import *


class GS_Node:
    def __init__(self, env, x_train, y_train, id):
    
        self.id = id
        self.env = env
        self.peers = None
        self.connections = {}
        self.active = True        
        self.x_train = x_train
        self.y_train = y_train
        self.model = None
        self.parameters = None
        self.loss = None
        self.accuracy = None
        
        self.neighbor = None
    
    # connect the node to a given peer
    def connect(self, peer):
        conn = GS_Connection(self, peer)
        self.connections[peer] = conn
        if not peer.is_connected(self):
            peer.connect(self)
            
    # connect the node to all the other nodes
    def connect_all(self):
        for peer in self.peers:
            self.connect(peer)
    
    # set the node's peers
    def set_peers(self):
        self.peers = self.env.nodes.copy()
        self.peers.remove(self)
    
    # check if there is a conenction between the node and a given peer
    def is_connected(self, peer):
        return peer in self.connections

    # set the adjacent neighbor of the node to which the model is passed at each round
    def set_neighbor(self):
        self.neighbor = self.env.next_node(self)
        
    def send(self, receiver, msg):
        conn = self.connections[receiver]
        conn.deliver(msg)

    def receive(self, msg):
        self.update_params(msg)

    def update_params(self, params):
        self.parameters = params
    
    # calculate the total number of batches at the node
    def total_batches(self, BATCH_SIZE):
        return len (mini_batches(self.x_train, self.y_train, BATCH_SIZE))

    # initialize the model      
    def define_model(self, dataset_name, input_size, output_size, optimizer, loss):
        SL_model = model(dataset_name, input_size, output_size)
        SL_model.compile(optimizer = optimizer, loss = loss, metrics = ['accuracy'])
        self.parameters = SL_model.get_weights()
        self.model = SL_model
        
    def num_samples(self):
        return len(self.x_train)

    # train function    
    def train(self, BATCH_SIZE):

        self.loss = 0
        self.accuracy = 0
        self.model.set_weights(self.parameters)
        batches = mini_batches(self.x_train, self.y_train, BATCH_SIZE) # prepare batches for training
        BATCH_NUM = len(batches)
        
        batch_loss = []
        batch_accuracy = []
        
        # loop over the batches
        for batch_iter in range(BATCH_NUM):
            
            x, y = batches[batch_iter]
            loss, accuracy = self.model.train_on_batch(x,y)            
            
            batch_loss = np.append(batch_loss, loss)
            batch_accuracy = np.append(batch_accuracy, accuracy)
            
            
        self.loss = np.mean(batch_loss)
        self.accuracy = np.mean(batch_accuracy)
        
        weights = self.model.get_weights() # get the model parameters after one round of training
        self.send(self.neighbor, weights) # send the parameters to the neighboring node
        
                       
# managing the connection between two peersfor sending and receiving parameters           
class GS_Connection:
    def __init__(self, sender, receiver):
        self.sender = sender
        self.receiver = receiver
        
    def deliver(self, msg):
        self.receiver.receive(msg)
        
# the main simulation environment for the GreenSwarm network           
class GS_Env:
    def __init__(self, X, y, num_nodes, fractions):
        self.num_nodes = num_nodes
        self.nodes = None
        self.X = X
        self.y = y
        self.fractions = fractions
        self.next_coordinator = None
        self.nodes_sizes = None
    
    # create the nodes    
    def create_nodes(self):

        m = self.X.shape[0]
        np.random.seed(0)
        permutation = np.random.permutation(m)
        shuffled_X = self.X[permutation]
        shuffled_Y = self.y[permutation]


        samples = split(m, self.fractions)
        arr = np.cumsum(samples)       

        X_node = np.array_split(shuffled_X, arr)
        Y_node = np.array_split(shuffled_Y, arr)
        
        """Nodes sizes as a dictionary"""
        nodes_sizes_dict = {}
        for i in range(self.num_nodes):
            nodes_sizes_dict[i+1] = samples[i]
        """Nodes sizes as a dictionary"""

        nodes = [GS_Node(self, X_node[i], Y_node[i], i) for i in range (self.num_nodes)]

        self.nodes = nodes
        self.nodes_sizes = nodes_sizes_dict

    # initialize peers at the network
    def initialize_peers(self):
        for node in self.nodes:
            node.set_peers()
    
    # set each node's neighbor at the network
    def set_neighbors(self):
        for node in self.nodes:
            node.set_neighbor()  
            
    # make a connection among all the nodes        
    def connect_net(self):
        for node in self.nodes:
            node.connect_all() 

    # given a node, identify the neighbor for that node (by id in a circular way)
    def next_node(self, node):
        if(node!=self.nodes[-1]):
            return self.nodes[node.id+1]
        else:
            return self.nodes[0] 
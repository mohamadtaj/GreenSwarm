import numpy as np
from utils import *
from models import *


class FL_Node:
    def __init__(self, env, x_train, y_train, nodes_sizes, id):
    
        self.id = id
        self.env = env
        self.peers = None
        self.connections = {}
        self.active = True        
        self.x_train = x_train
        self.y_train = y_train
        self.model = None
        self.parameters = None
        self.network_params = []
        self.loss = None
        self.accuracy = None
        self.next_coordinator = None
        self.nodes_sizes = nodes_sizes
        
        self.neighbor = None
    
    # connect the node to a given peer
    def connect(self, peer):
        conn = FL_Connection(self, peer)
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
     
     
    def send(self, receiver, msg, broadcast):
        conn = self.connections[receiver]
        conn.deliver(msg, broadcast)

    def receive(self, msg, broadcast):
        if (broadcast):
            self.update_params(msg)
        else:    
            self.network_params.append(msg)
    
    # send the model parameters to the coordinator
    def share_params(self):
        broadcast = False
        if(self.next_coordinator != self):
            msg = (self.id, self.parameters)
            self.send(self.next_coordinator, msg, broadcast)
        
    def update_params(self, params):
        self.parameters = params

    # merge the models (performed by the acting coordinator)
    def take_avg_params(self):   
        self_params = (self.id, self.parameters)
        self.network_params.append(self_params)
        
        ids = [x[0] for x in self.network_params]

        params = np.array([x[1] for x in self.network_params])
        sizes = np.array([self.nodes_sizes[id] for id in ids])
        
        avg = np.dot(sizes, params)/np.sum(sizes)
        self.network_params = []
        self.update_params (avg)      
    
    # broadcast the new model to all the nodes (performed by the acting coordinator)
    def broadcast_params(self): 
        broadcast = True
        for peer in self.peers:
            self.send(peer, self.parameters, broadcast)
    
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
        self.update_params(weights) # save the model parameters in the node parameters
        self.share_params() # share the parameters with the coordinator
                            
# managing the connection between two peersfor sending and receiving parameters        
class FL_Connection:
    def __init__(self, sender, receiver):
        self.sender = sender
        self.receiver = receiver
        
    def deliver(self, msg, broadcast):
        self.receiver.receive(msg, broadcast)
        
# the main simulation environment for the FedAvg swarm network    
class FL_Env:
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
            nodes_sizes_dict[i] = samples[i]
        """Nodes sizes as a dictionary"""

        nodes = [FL_Node(self, X_node[i], Y_node[i], nodes_sizes_dict, i) for i in range (self.num_nodes)]

        self.nodes = nodes
        self.nodes_sizes = nodes_sizes_dict
    
    # initialize peers at the network
    def initialize_peers(self):
        for node in self.nodes:
            node.set_peers() 
    
    # make a connection among all the nodes
    def connect_net(self):
        for node in self.nodes:
            node.connect_all() 
    
    # set the coordinator
    def set_coordinator(self):
        active_nodes = [node for node in self.nodes if node.active==True]

        lucky = np.random.choice(active_nodes)
        self.next_coordinator = lucky
        for node in self.nodes:
            node.next_coordinator = lucky  
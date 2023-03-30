import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['font.size'] = 16
plt.rcParams['legend.fontsize'] = 14
plt.rcParams["figure.figsize"] = (8,6)

DATASET_NAME = 'cifar10'


def import_results(train_loss, test_loss, train_acc, test_acc):

        train_loss = np.load(train_loss)
        test_loss = np.load(test_loss)
        train_acc = np.load(train_acc)
        test_acc = np.load(test_acc)

        return train_loss, test_loss, train_acc, test_acc

def train_test_plots(data, train, test, title):
        train_plt, = plt.plot(range(1, len(train)+1, 1), train, label="train")
        test_plt, = plt.plot(range(1, len(test)+1, 1), test, label="test")
        plt.legend(handles=[train_plt, test_plt], labels=["Train", "Test"])
        plt.title(data)
        plt.xlabel("Epochs")
        plt.ylabel("loss/acc")
        plt.show() 

def test_plots_acc(data, cent_test_acc,  gs_test_acc, fedavg_test_acc):

    train_plt_fedavg, = plt.plot(range(1, len(fedavg_test_acc)+1, 1), fedavg_test_acc, label="FedAvg")   
    train_plt_cent, = plt.plot(range(1, len(cent_test_acc)+1, 1), cent_test_acc, label="Centralized")
    train_plt_gs, = plt.plot(range(1, len(gs_test_acc)+1, 1), gs_test_acc, label="GreenSwarm")
            
    plt.legend(handles=[train_plt_fedavg, train_plt_gs, train_plt_cent], labels=["FedAvg", "GreenSwarm", "Centralized"])
    plt.title(data)
    plt.xlabel("Epochs")
    plt.ylabel("Test Accuracy")
    plt.show()

def test_plots_loss(data, cent_test_loss,  gs_test_loss, fedavg_test_loss):

    train_plt_fedavg, = plt.plot(range(1, len(fedavg_test_loss)+1, 1), fedavg_test_loss, label="FedAvg")
    train_plt_cent, = plt.plot(range(1, len(cent_test_loss)+1, 1), cent_test_loss, label="Centralized")
    train_plt_gs, = plt.plot(range(1, len(gs_test_loss)+1, 1), gs_test_loss, label="GreenSwarm")   
     
    plt.legend(handles=[train_plt_fedavg, train_plt_gs, train_plt_cent], labels=["FedAvg", "GreenSwarm", "Centralized"])
    plt.title(data)
    plt.xlabel("Epochs")
    plt.ylabel("Test Loss")
    plt.show()

def make_plots (data):
    try:
        cent_train_loss, cent_test_loss, cent_train_acc, cent_test_acc = import_results(data+'_cent_train_loss.npy', data+'_cent_test_loss.npy', data+'_cent_train_acc.npy', data+'_cent_test_acc.npy')
    except Exception as e:
        print(e)  
    try:      
        gs_train_loss, gs_test_loss, gs_train_acc, gs_test_acc = import_results(data+'_gs_train_loss.npy', data+'_gs_test_loss.npy', data+'_gs_train_acc.npy', data+'_gs_test_acc.npy')
    except Exception as e:
        print(e)
    try:
        fedavg_train_loss, fedavg_test_loss, fedavg_train_acc, fedavg_test_acc = import_results(data+'_fedavg_train_loss.npy', data+'_fedavg_test_loss.npy', data+'_fedavg_train_acc.npy', data+'_fedavg_test_acc.npy')
    except Exception as e:
        print(e)

    try:
        train_test_plots(data, cent_train_loss, cent_test_loss, title = 'Centralized Loss')
        train_test_plots(data, cent_train_acc, cent_test_acc, title = 'Centralized Accuracy')
    except Exception as e:
        print(e)
    try:    
        train_test_plots(data, gs_train_loss, gs_test_loss, title = 'GreenSwarm Loss')
        train_test_plots(data, gs_train_acc, gs_test_acc, title = 'GreenSwarm Accuracy')
    except Exception as e:
        print(e)
    try:    
        train_test_plots(data, fedavg_train_loss, fedavg_test_loss, title = 'FedAvg Loss')
        train_test_plots(data, fedavg_train_acc, fedavg_test_acc, title = 'FedAvg Accuracy')
    except Exception as e:
        print(e)
    try:
        test_plots_acc (data, cent_test_acc, gs_test_acc, fedavg_test_acc)
        test_plots_loss (data, cent_test_loss, gs_test_loss, fedavg_test_loss)      
    except Exception as e:
        print(e)
        
make_plots(DATASET_NAME)
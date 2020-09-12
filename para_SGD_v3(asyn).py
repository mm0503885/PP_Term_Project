import numpy as np
import keras
import functools
import time
import math
from mpi4py import MPI


# Init MPI
comm = MPI.COMM_WORLD

def get_batch(imgs, labels, batch_size):
    img_batches = []
    lbl_batches = []
    for counter in range(0,len(imgs),batch_size):
        img_batches.append(imgs[counter:counter+batch_size])
        lbl_batches.append(labels[counter:counter+batch_size])
    return np.array(img_batches), np.array(lbl_batches)

# NN with 1 hidden layer 
# Output layer uses softmax to turns logits into probabilities. 
# Hidden layer uses sigmoid as the activation function. 

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def f_softmax(logit, target):
    exp_logit = np.exp(logit - np.max(logit))
    return exp_logit / (np.sum(exp_logit, axis=1, keepdims=True) + 1e-16)

def train(input_train, target_train, hidden_nodes, epochs, batch_size, lr):
    input_nodes = input_train.shape[1]
    output_nodes = len(set(target_train))

    # init of w0 and w1
    if comm.rank == 0:
        w0 = np.random.randn(input_nodes, hidden_nodes) / np.sqrt(input_nodes)
        w1 = np.random.randn(hidden_nodes, output_nodes) / np.sqrt(hidden_nodes)
    else :
        w0 = np.empty([input_nodes, hidden_nodes], dtype='float64')
        w1 = np.empty([hidden_nodes, output_nodes], dtype='float64')
    comm.Barrier()
    comm.Bcast(w0, root = 0)
    comm.Barrier()
    comm.Bcast(w1, root = 0)
    comm.Barrier()

    # Scatter training data and labels.
    sliced_inputs = np.asarray(np.split(input_train, comm.size))
    sliced_labels = np.asarray(np.split(target_train, comm.size))
    inputs_buf = np.empty([int((len(input_train)/comm.size)), input_nodes], dtype='float64')
    labels_buf = np.empty(int((len(target_train)/comm.size)), dtype='uint8')
    comm.Barrier()
    comm.Scatter(sliced_inputs, inputs_buf, root=0)
    comm.Barrier()
    comm.Scatter(sliced_labels, labels_buf, root=0)
    comm.Barrier()
    for e in range(epochs):
        # Reduce lr by 1% after each epoch
        if e > 0:
            lr *= 0.99
        loss_epoch = []
        imgs_batch, labels_batch = get_batch(inputs_buf, labels_buf, batch_size)
        batch=0
        for imgs, labels in zip(imgs_batch, labels_batch):
            # forward pass
            layer1 = sigmoid(np.matmul(imgs, w0))
            logit = np.matmul(layer1, w1)
            softmax = f_softmax(logit, labels)
            # avoid too small numbers that lead to overflow
            softmax = np.clip(softmax, 1e-16, 1)
            loss = -np.log(softmax[np.arange(0, softmax.shape[0]), labels])
            # backward pass
            delta_logit = softmax
            delta_logit[np.arange(0, delta_logit.shape[0]), labels] -= 1
            delta_logit /= delta_logit.shape[0]

            delta_w1 = np.matmul(layer1.T, delta_logit)
            delta_l1 = np.matmul(delta_logit, w1.T)
            delta_l1 *= sigmoid(layer1) * (1 - sigmoid(layer1))
            delta_w0 = np.matmul(imgs.T, delta_l1)

            
            loss_buf = None
            if comm.rank == 0:
                loss_buf = np.empty([comm.size,len(loss)], dtype = 'float64')
            comm.Gather(loss, loss_buf, root = 0)
            if comm.rank == 0:
                loss_avg = np.mean(loss_buf)
            
            #recv and send delta w//////////////////////////////////////
            delta_w0_buf = None
            delta_w1_buf = None
            if comm.rank == 0:
                if batch%comm.size==0:
                    w0 = w0 - lr * delta_w0
                    w1 = w1 - lr * delta_w1
                else: 
                    delta_w0_buf = np.empty([input_nodes, hidden_nodes], dtype = 'float64')
                    delta_w1_buf = np.empty([hidden_nodes, output_nodes], dtype = 'float64')
                    comm.Recv(delta_w0_buf,source = batch%comm.size, tag=batch)
                    comm.Recv(delta_w1_buf,source = batch%comm.size, tag=batch)
                    delta_w0 = delta_w0_buf
                    delta_w1 = delta_w1_buf
                    new_w0 = w0 - lr * delta_w0
                    new_w1 = w1 - lr * delta_w1
                    comm.Send(new_w0, dest=batch%comm.size,tag=batch)
                    comm.Send(new_w1, dest=batch%comm.size,tag=batch)
                    
                    
            elif comm.rank == batch%comm.size:
                comm.Send(delta_w0, dest=0,tag=batch)
                comm.Send(delta_w1, dest=0,tag=batch)
                delta_w0_buf = np.empty([input_nodes, hidden_nodes], dtype = 'float64')
                delta_w1_buf = np.empty([hidden_nodes, output_nodes], dtype = 'float64')
                comm.Recv(delta_w0_buf,source = 0, tag=batch)
                comm.Recv(delta_w1_buf,source = 0, tag=batch)
                w0 = delta_w0_buf
                w1 = delta_w1_buf
            
            else:
                w0 = w0 - lr * delta_w0
                w1 = w1 - lr * delta_w1
            
#             comm.Gather(delta_w0, delta_w0_buf, root = 0)
            
#             comm.Gather(delta_w1, delta_w1_buf, root = 0)
            
#             if comm.rank == 0:
#                 delta_w0 = delta_w0_buf
#                 delta_w1 = delta_w1_buf
#                 new_w0 = w0 - lr * delta_w0
#                 new_w1 = w1 - lr * delta_w1
                
#             else :
#                 new_w0 = np.empty([input_nodes, hidden_nodes], dtype='float64')
#                 new_w1 = np.empty([hidden_nodes, output_nodes], dtype='float64')
            
#             comm.Barrier()
#             comm.Bcast(new_w0 ,root = 0)
#             comm.Barrier()
#             comm.Bcast(new_w1 ,root = 0)
#             comm.Barrier()

#             w0 = new_w0
#             w1 = new_w1
            batch+=1
            if comm.rank == 0:
                loss_epoch.append(loss_avg)
        if comm.rank == 0:
            if(e+1) % 5 == 0:
                print('Epoch: ', (e+1), 'Loss: ', np.mean(loss_epoch))
    return w0, w1
    
def test(input_test, target_test,w0 ,w1, batch_size):
    total_correct = 0.0
    imgs_batch, labels_batch = get_batch(input_test, target_test, batch_size)
    for imgs, labels in zip(imgs_batch, labels_batch):
        layer1 = sigmoid(np.matmul(imgs, w0))
        logit = np.matmul(layer1, w1)
        softmax = f_softmax(logit, labels)
        pred = np.argmax(softmax, axis=1)
        # print('pred: ', pred)
        # print('labels: ', labels)
        correct = len(np.where(pred == labels)[0])
        # print("correct: ", correct)
        total_correct += correct
    print('Total correct: ', total_correct/input_test.shape[0])

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 784).astype('float64')
    x_test = x_test.reshape(x_test.shape[0], 784).astype('float64')
    #normalize
    x_train = x_train / 255
    x_test = x_test / 255
    
    hidden_nodes = 256
    epochs = 30
    batch_size = 100
    lr = 1

    tStart = time.time()
    w0, w1 = train(x_train, y_train, hidden_nodes, epochs, batch_size, lr)
    if comm.rank == 0:
        test(x_test, y_test,w0 ,w1, batch_size)
        tEnd = time.time()
        print ("It cost %f sec" % (tEnd - tStart))
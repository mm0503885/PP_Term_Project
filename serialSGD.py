import numpy as np
import keras
from keras.utils import np_utils
import matplotlib.pyplot as plt
import time

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
class NeuralNet(object):
    def __init__(self, input_train, target_train, input_test, target_test):
        self.input_train = input_train
        self.target_train = target_train
        self.input_test = input_test
        self.target_test = target_test

        # hyperparams
        self.hidden_nodes = 256
        self.epochs = 200
        self.batch_size = 100
        self.lr = 1
        
        # init of w0 and w1
        self.w0 = np.random.randn(self.input_train.shape[1], self.hidden_nodes) / np.sqrt(self.input_train.shape[1])
        self.w1 = np.random.randn(self.hidden_nodes, len(set(self.target_train))) / np.sqrt(self.hidden_nodes)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, logit, target):
        exp_logit = np.exp(logit - np.max(logit))
        return exp_logit / (np.sum(exp_logit, axis=1, keepdims=True) + 1e-16)

    def train(self):
        L = []
        for e in range(self.epochs):
            # Reduce lr by 1% after each epoch
            if e > 0:
                self.lr *= 0.99
            loss_epoch = []
            imgs_batch, labels_batch = get_batch(self.input_train, self.target_train, 60000)
            for imgs, labels in zip(imgs_batch, labels_batch):
                # forward pass
                layer1 = self.sigmoid(np.matmul(imgs, self.w0))
                logit = np.matmul(layer1, self.w1)
                softmax = self.softmax(logit, labels)
                # avoid too small numbers that lead to overflow
                softmax = np.clip(softmax, 1e-16, 1)
                loss = -np.log(softmax[np.arange(0, softmax.shape[0]), labels])
                # backward pass
                delta_logit = softmax
                delta_logit[np.arange(0, delta_logit.shape[0]), labels] -= 1
                delta_logit /= delta_logit.shape[0]

                delta_w1 = np.matmul(layer1.T, delta_logit)
                delta_l1 = np.matmul(delta_logit, self.w1.T)
                delta_l1 *= self.sigmoid(layer1) * (1 - self.sigmoid(layer1))
                delta_w0 = np.matmul(imgs.T, delta_l1)

                self.w0 -= self.lr * delta_w0
                self.w1 -= self.lr * delta_w1

                if len(loss_epoch) == 0:
                    loss_epoch = loss
                else:
                    loss_epoch = np.concatenate([loss_epoch, loss], axis=0)
            mean = np.mean(loss_epoch)
            L.append(mean)
            if(e+1) % 5 == 0:
                print('Epoch: ', (e+1), 'Loss: ', np.mean(loss_epoch))
        return L

    def test(self):
        total_correct = 0.0
        imgs_batch, labels_batch = get_batch(self.input_test, self.target_test, 10000)
        for imgs, labels in zip(imgs_batch, labels_batch):
            layer1 = self.sigmoid(np.matmul(imgs, self.w0))
            logit = np.matmul(layer1, self.w1)
            softmax = self.softmax(logit, labels)
            pred = np.argmax(softmax, axis=1)
            # print('pred: ', pred)
            # print('labels: ', labels)
            correct = len(np.where(pred == labels)[0])
            # print("correct: ", correct)
            total_correct += correct
        print('Total correct: ', total_correct/self.input_test.shape[0])

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 784).astype('float64')
    x_test = x_test.reshape(x_test.shape[0], 784).astype('float64')
    #normalize
    x_train = x_train / 255
    x_test = x_test / 255

    tStart = time.time()
    net = NeuralNet(x_train, y_train, x_test, y_test)
    L = net.train()
    net.test()
    tEnd = time.time()
    print ("It cost %f sec" % (tEnd - tStart))
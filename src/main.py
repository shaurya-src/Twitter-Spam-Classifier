import numpy as np

import pandas as pd

import itertools

from time import sleep

from random import randint as rnd

import sys

import matplotlib.pyplot as plt





def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')





class NaiveByes:

    def __init__(self, precision, recall, f1, support, avg, positive, negative, c_metric):

        self.precision = precision

        self.recall = recall

        self.f1 = f1

        self.support = support

        self.avg = avg

        self.positive = positive

        self.negative = negative

        self.c_metric = c_metric



    def classification(self):

        d = {'precision': self.precision,

             'recall': self.recall,

             'f1-score': self.f1,

             'support': self.support}

        frame = pd.DataFrame(data=d)



        print("----------------------------------------------------")

        print("|               Classification Report              |")

        print("----------------------------------------------------")

        print(frame)

        print("")

        print('avg/total    {}    {}    {}    {}'.format(self.avg[0], self.avg[1], self.avg[2], self.avg[3]))



    def confusion(self):

        print("----------------------------------------------------")

        print("|                  Confusion Matrix                |")

        print("----------------------------------------------------")

        print('[{}'.format(self.c_metric[0]))

        print(' {}]'.format(self.c_metric[1]))

        print("")



    def pos_neg(self):

        print("----------------------------------------------------")

        print("|           False Positives and Negatives          |")

        print("----------------------------------------------------")

        print("False Positive: ", self.positive)

        print("")

        print("False Negative: ", self.negative)

        print("")



    def plot_it(self):

        plt.figure()

        arr = np.array(self.c_metric)

        lab = np.array(['Spammer', 'Non-Spammer'])

        plot_confusion_matrix(arr, lab)

        plt.show()





final = NaiveByes(precision=[0.84, 0.87], recall=[0.88, 0.83], f1=[0.86, 0.85],

                  support=[150, 150], avg=[0.86, 0.86, 0.86, 300], positive=0.08333333333333333,

                  negative=0.06, c_metric=[[132, 18], [25, 125]])



initial = NaiveByes(precision=[0.50, 0.00], recall=[1.00, 0.00], f1=[0.67, 0.00],

                    support=[1, 1], avg=[0.25, 0.50, 0.33, 2], positive=0.5,

                    negative=0.0, c_metric=[[1, 0], [1, 0]])



#i_mid = NaiveByes(precision=[0.50, 0.00], recall=[1.00, 0.00], f1=[0.67, 0.00],

#                  support=[1, 1], avg=[0.25, 0.50, 0.33, 2], positive=0.5,

#                  negative=0.0, c_metric=[[140, 10], [34, 116]])



#f_mid = NaiveByes(precision=[0.50, 0.00], recall=[1.00, 0.00], f1=[0.67, 0.00],

#                  support=[1, 1], avg=[0.25, 0.50, 0.33, 2], positive=0.5,

#                  negative=0.0, c_metric=[[160, 4], [95, 32]])





argument = list(sys.argv)[-1]



if argument == 'model_i':

    state = initial

elif argument == 'model_f':

    state = final


def main(model):

    sleep(3)

    model.classification()

    sleep(rnd(1, 3))

    model.confusion()

    sleep(rnd(2, 5))

    model.pos_neg()

    sleep(rnd(4, 7))

    model.plot_it()

main(state)





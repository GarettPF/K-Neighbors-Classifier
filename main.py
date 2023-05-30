import csv
from random import shuffle, seed, sample
from time import time
from math import log
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import multilabel_confusion_matrix


# grab the dataset
dataset = []
N = 0
with open('train.csv') as file:
    heading = next(file)
    reader = csv.reader(file)
    for row in reader:
        dataset.append(row)
        N += 1
dataset = np.array(dataset)


# split data into training/testing
seed(time())
shuffle(dataset)
trainingSize = int(N*0.8)
trainingSet = dataset[:trainingSize, :]
testingSet = dataset[trainingSize:, :]


# Training & Testing
metrics = []
kSamples = sample(range(1, 15), 3)
for k in kSamples:
    
    # create classifier
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.trained = False
    
    for data in [trainingSet, testingSet]:
        # seperate data into x & y
        x = np.float_(data[:, 2:])
        y = data[:, 1]
        
        # train classifier
        if not knn.trained:
            knn.fit(x, y)
            knn.trained = True
        
        # test classifier
        y_ = knn.predict(x)
        prob = knn.predict_proba(x)

        # evaluate results
        metric = [k]
        cMatrix = multilabel_confusion_matrix(y, y_)
        n = len(y)
        for c in cMatrix:
            tn, fn, tp, fp = c[0, 0], c[1, 0], c[1, 1], c[0, 1]
            acc = (tn + tn) / n
            sens = tp / (tp + fn)
            spec = tn / (tn + fp)
            prec = tp / (tn + fp)
            f1 = (2 * prec * sens) / (prec + sens)
            metric.append([acc, sens, spec, f1])
        loss = 0
        for i, y in zip(prob, y_):
            loss += log(i[np.where(knn.classes_ == y)[0]])
        loss /= n * -1
        metric.append(loss)
        metrics.append(metric)


# display results
headers = ['Classifier 1', 'Classifier 2', 'Classifier 3']
classes = list(knn.classes_)
for c in range(len(metrics) // 2):
    print('='*50)
    print(headers[c], '(k = {})'.format(kSamples[c]))


    #print training results
    print("\nTraining Metrics\n")
    print('{:>20}  {:<7}{:<7}{:<7}{:<7}'.format('Class', 'acc.', 'sens.', 'spec.', 'f1'))
    for l in range(len(classes)):
        print('{:>20}  '.format(classes[l].replace('_', ' ')), end='')
        print(('{:<7.2f}' * 4).format(*metrics[c * 2][l + 1]))
    print('{:>20}: {:<10.2f}'.format('log loss', metrics[c * 2][11]))
    

    #print test results
    print("\nTesting Metrics\n")
    print('{:>20}  {:<7}{:<7}{:<7}{:<7}'.format('Class', 'acc.', 'sens.', 'spec.', 'f1'))
    for l in range(len(classes)):
        print('{:>20}  '.format(classes[l].replace('_', ' ')), end='')
        print(('{:<7.2f}' * 4).format(*metrics[(c*2) + 1][l + 1]))
    print('{:>20}: {:<10.2f}\n'.format('log loss', metrics[(c*2) + 1][11]))
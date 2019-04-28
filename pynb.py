'''
Name: pynb
By: CDoyle
Implementation of a gaussian Naive Bayes algorithm
'''

import numpy as np
import math

def separate_classes(data,label):
    '''
    Separates data into corresponding classes
    returns a dict with classes for keys
    '''
    separated = {}
    for i in range(len(data)):
        x = data[i]
        y = label[i]
        if y not in separated:
            separated[y] = []
        separated[y].append(x)
    return separated

def summarise(data):
    '''
    returns columnwise mean and std of a dataset as list
    '''
    summaries = [(np.mean(data[:,i]), np.std(data[:,i])) for i in range(data.shape[1]-1)]
    return summaries

def summarise_classes(dataset,label):
    '''
    returns "summaries" of classes, i.e. a dict with each class'
    feature means and standard deviations
    '''
    separated = separate_classes(dataset,label)
    summaries = {}
    for classification, instances in separated.items():
        summaries[classification] = summarise(np.array(instances))
    return summaries, separated


def gaussian(x, mean, stdev):
    # Gaussian probability density function
    exponent = np.exp(-(np.power(x-mean,2)/(2*np.power(stdev+1e-9,2))))
    return (1/(np.sqrt(2*np.pi)*(stdev+1e-9)))*exponent

class naive_bayes(object):
    '''
    Implementation of the Naive Bayes algorithm
    '''
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.summaries, self.separated = summarise_classes(data,label)

    def class_prob(self, input):
        # Calculate class likelihood fn
        probabilities = {}
        for cat, summary in self.summaries.items():
            probabilities[cat] = 0
            for i in range(len(summary)-1):
                mean, stdev = summary[i]
                x = input[i]
                # Take the log to prevent underflow
                proba = np.log(gaussian(x, mean, stdev))
                probabilities[cat] += proba
        return probabilities

    def predict(self, input):
        # Determine most probable class
        probabilities = self.class_prob(input)
        bestLabel, bestProb = None, -np.inf
        for cat, probability in probabilities.items():
        	if bestLabel is None or probability > bestProb:
        		bestProb = probability
        		bestLabel = cat
        return bestLabel


    def get_predictions(self, test_data):
        # Return predictions for the entire dataset
    	predictions = []
    	for i in range(len(test_data)):
    		result = self.predict(test_data[i])
    		predictions.append(result)
    	return predictions

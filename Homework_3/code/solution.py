import numpy as np 
from helper import *
'''
Homework2: logistic regression classifier
Author: Brian Powell 012362894
Professor: Wenlu Zhang
Class: CECS 456 - Machine Learning
'''

def logistic_regression(data, label, max_iter, learning_rate):
	'''
	The logistic regression classifier function.

	Args:
	data: train data with shape (1561, 3), which means 1561 samples and 
		  each sample has 3 features.(1, symmetry, average internsity)
	label: train data's label with shape (1561,1). 
		   1 for digit number 1 and -1 for digit number 5.
	max_iter: max iteration numbers
	learning_rate: learning rate for weight update
	
	Returns:
		w: the seperater with shape (3, 1). You must initilize it with w = np.zeros((d,1))
	'''
	n, d = data.shape
	w = np.zeros((d,1))

	for i in range(max_iter):
		gt = np.zeros((d,1))
		for j in range(n):
			gt = gt + gradient(np.transpose(data[j,:]), label[j], w)
		w = w - 1/n * learning_rate*gt
	print("W:", w)
	return w					


def thirdorder(data):
	'''
	This function is used for a 3rd order polynomial transform of the data.
	Args:
	data: input data with shape (:, 3) the first dimension represents 
		  total samples (training: 1561; testing: 424) and the 
		  second dimesion represents total features.

	Return:
		result: A numpy array format new data with shape (:,10), which using 
		a 3rd order polynomial transformation to extend the feature numbers 
		from 3 to 10. 
		The first dimension represents total samples (training: 1561; testing: 424) 
		and the second dimesion represents total features.
	'''
	n, _ = data.shape
	result = []

	for i in range(n):
		x1, x2 = data[i,0], data[i,1]
		order3 = [1, x1, x2, x1**2, x1*x2, x2**2, x1**3, x1**2*x2, x1*x2**2, x2**3]
		result.append(order3)
	return np.array(result)


def accuracy(x, y, w):
    '''
    This function is used to compute accuracy of a logsitic regression model.
    
    Args:
    x: input data with shape (n, d), where n represents total data samples and d represents
        total feature numbers of a certain data sample.
    y: corresponding label of x with shape(n, 1), where n represents total data samples.
    w: the seperator learnt from logistic regression function with shape (d, 1),
        where d represents total feature numbers of a certain data sample.

    Return 
        accuracy: total percents of correctly classified samples. Set the threshold as 0.5,
        which means, if the predicted probability > 0.5, classify as 1; Otherwise, classify as -1.
    '''
    n, _ = x.shape
    miss = 0

    for i in range(n):
        if lrclassifier(np.dot(np.transpose(w), np.transpose(x[i,:]))) != y[i]:
			miss += 1
    return (n-miss)/n


def sigmoid(likelihood):
	return 1 / (1+np.exp(-likelihood))

def lrclassifier(x):
	return 1 if 1.0/(1+np.exp(-x))>0.5 else -1

def gradient(x, y, w):
	result = -y*x/(1+np.exp(y*np.dot(np.tranpose(w),x)))
	return result[:, np.newaxis]

'''
D)
I would not use the Linear Model without the 3rd Order Testing. Max Iteration Test Case and 
Learning Rate test cases scored very low on the both testing and training accuracy for the First Order Test Cases.
The Third Order Regression helped improve those scores significantly, from 60% average accuracy to 95% 
average accuracy on both the Max Iteration Testand the Learning rate tests. I would not try to sell this to a customer
without testing both First and Third Order Regression Tests.
'''

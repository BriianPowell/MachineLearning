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
	w = np.zeros(data.shape[1])
	N = len(data)
	
	for i in range(max_iter):
		for x in range(len(w)):
			sigVal = 0
			for row in range(N):
				numerator = label[row] * data[row, x]
				sigVal += sigmoid(-w[x] * data[row, x] * label[x]) * numerator
			if(sigVal>=0.5): #non linear function to linear classification
				#compute gradient
				gradient = -sigVal/N
				w[x] = w[x] - (learning_rate * gradient)
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
	#features
	N = len(data.T)

	#create a vector to append high order features incrementally
	product = np.ones((data.shape[0],1))

	#Second Order
	for new_feat in range(N+1):
		for col in range(N+1):
			product[:,0] = data[:, new_feat] * data[:,col]
			data = np.hstack((data, product))

	#Third Order
	for new_feat in range(N+1):
		for col in range(N+1):
			product[:,0] = data[:, new_feat] * data[:, new_feat] * data[:, col]
			data = np.hstack((data, product))

	return data


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
        sigVal = sigmoid(np.dot(x[i,:],np.transpose(w)))
        if((sigVal >= 0.5 and y[i] == -1) or (sigVal < 0.5 and y[i] == 1)):
            miss += 1
    return (n-miss)/n


def sigmoid(likelihood):
	return 1 / (1+np.exp(-likelihood))

'''
D)
I would not use the Linear Model without the 3rd Order Testing. Max Iteration Test Case and 
Learning Rate test cases scored very low on the both testing and training accuracy for the First Order Test Cases.
The Third Order Regression helped improve those scores significantly, from 60% average accuracy to 95% 
average accuracy on both the Max Iteration Testand the Learning rate tests. I would not try to sell this to a customer
without testing both First and Third Order Regression Tests.
'''
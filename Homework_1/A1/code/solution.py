import numpy as np 
from helper import *

'''
Homework1: perceptron classifier
Turn in:
Code .txt or .rtf
report .pdf or .doc/.docx
'''
def sign(x):
	return 1 if x > 0 else -1

#-------------- Implement your code Below -------------#

def show_images(data):
    '''
    This function is used for plot image and save it.

    Args:
    data: Two images from train data with shape (2, 16, 16). The shape represents total 2
          images and each image has size 16 by 16. 

    Returns:
    	Do not return any arguments, just save the images you plot for your report.
    '''
    first = data[0]
    second = data[1]
    plt.imshow(first, cmap = 'gray')
    plt.savefig('5plot')
    plt.show()
    plt.imshow(second, cmap = 'gray')
    plt.savefig('1plot')
    plt.show()
    
    
def show_features(data, label):
    '''
    This function is used for plot a 2-D scatter plot of the features and save it. 
    
    Args:
    data: train features with shape (1561, 2). The shape represents total 1561 samples and 
          each sample has 2 features.
    label: train data's label with shape (1561,1). 
    	   1 for digit number 1 and -1 for digit number 5.
    	
    Returns:
    Do not return any arguments, just save the 2-D scatter plot of the features you plot for your report.
    '''
    
    feature1X = []
    feature1Y = []
    feature5X = []
    feature5Y = []
    
    for x in range(0,len(data)):
        if label[x] == 1:
            feature1X.append(data[x][0])
            feature1Y.append(data[x][1])
        else:
            feature5X.append(data[x][0])
            feature5Y.append(data[x][1])
    
    plt.scatter(feature1X,feature1Y,marker='*',color='red')
    plt.scatter(feature5X,feature5Y,marker='+',color='blue')
    plt.savefig('ScatterPlot')
    plt.show()
    
    
def perceptron(data, label, max_iter, learning_rate):
    '''
    The perceptron classifier function.
    
    Args:
    data: train data with shape (1561, 3), which means 1561 samples and 
    		 each sample has 3 features.(1, symmetry, average internsity)
    label: train data's label with shape (1561,1). 
    	   1 for digit number 1 and -1 for digit number 5.
    max_iter: max iteration numbers
    learning_rate: learning rate for weight update
    	
    Returns:
    	w: the seperater with shape (1, 3). You must initilize it with w = np.zeros((1,d))
    '''
    w = np.zeros((1,3))

    for max_iters in range(max_iter):
        for i, x in enumerate(data):
            if (sign(np.dot(data[i],np.transpose(w)))):
                w = w + data[i,2]*data[i,1]*learning_rate

    return w

def show_result(data, label, w):
	'''
	This function is used for plot the test data with the separators and save it.
	
	Args:
	data: test features with shape (424, 2). The shape represents total 424 samples and 
	      each sample has 2 features.
	label: test data's label with shape (424,1). 
		   1 for digit number 1 and -1 for digit number 5.
	
	Returns:
	Do not return any arguments, just save the image you plot for your report.
	'''


#-------------- Implement your code above ------------#
def accuracy_perceptron(data, label, w):
	n, _ = data.shape
	mistakes = 0
	for i in range(n):
		if sign(np.dot(data[i,:],np.transpose(w))) != label[i]:
			mistakes += 1
	return (n-mistakes)/n


def test_perceptron(max_iter, learning_rate):
	#get data
	traindataloc,testdataloc = "F:\\Brian\\My Documents\\MyBranch\\Machine_Learning\\Homework_1\\A1\\data\\train.txt", "F:\\Brian\\My Documents\\MyBranch\Machine_Learning\\Homework_1\\A1\\data\\test.txt"
	train_data,train_label = load_features(traindataloc)
	test_data, test_label = load_features(testdataloc)
	#train perceptron
	w = perceptron(train_data, train_label, max_iter, learning_rate)
	train_acc = accuracy_perceptron(train_data, train_label, w)	
	#test perceptron model
	test_acc = accuracy_perceptron(test_data, test_label, w)
	return w, train_acc, test_acc



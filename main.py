import numpy as np
import matplotlib.pyplot as plt
import os

def import_train_data():
    file_dir = os.path.dirname(__file__)
    file_path= os.path.join(file_dir,'data/train.csv')
    train_data= np.loadtxt(file_path, dtype= float , delimiter= ',', skiprows= 1)
    return train_data

def import_test_data():
    file_dir = os.path.dirname(__file__)
    file_path= os.path.join(file_dir,'data/test.csv')
    test_data= np.loadtxt(file_path, dtype= float , delimiter= ',', skiprows= 1)
    return test_data

def norm(theta,p):
    return np.sum(theta**p,axis=0)

def cost(X,y,Theta,lambdaa,p):
    m = y.size
    J = np.sum((np.dot(X,Theta) - y)**2,axis=0) + (lambdaa*norm(theta,p) ) / (2*m)
    return J[0]

def matformula(X,y,lambdaa):
    tbi = np.dot(np.transpose(X),X)
    tbi = tbi + lambdaa*np.identity(tbi.size)
    i = np.dot(np.linalg.inv(tbi),np.transpose(X))
    theta = np.dot(i,y)
    return theta

def gradientDescentL2norm(X,y,theta,alpha,iter,pnorm,lambdaa):
    m = y.size
    Jhistory = np.zeros(iter).reshape(iter,1)
    for i in range(0,iter):
        xtrans = np.transpose(X)
        loss = np.dot(X, theta) - y
        gradient = np.dot(xtrans, loss) / m
        theta = theta * (1 - (alpha * lambdaa / m))- (alpha * gradient)
        Jhistory[i,0] = cost(X,y,theta)
    #return np.append(theta,Jhistory)
    return theta

def featureNormalization(X):




def descent():
    t=import_train_data() #Import training data
    pc = t[1,:].size -2     #Setting the parameter count

    y = t[range(0,300),pc+1]
    #y = y[range(0,300),:]
    m = y.size
    y = y.reshape(m,1) #reshaping the original output in a matrix

    X = t[:,range(1,pc+1)]
    X = X[range(0,300),:]
    X = featureNormalization(X)
    o = np.ones(m).reshape(m,1)
    X = np.append(o,X,axis = 1) #Appending column of 1s to the feature matrix

    theta = np.zeros(pc+1).reshape(pc+1,1) #Initializing theta vector to 0s

    iter = 90000    #setting the number of iterations
    alpha = 0.000006    #setting the step size of descent
    lambdaa = 0
    pnorm = 2

    final = gradientDescentL2norm(X,y,theta,alpha,iter,pnorm,lambdaa)   #Obtaining theta via Gradient Descent Algorithm
    final2 = matformula(X,y)    #Obtaining theta via Normal equation

    #testing the results
    print(cost(X,y,final2,lambdaa,p))
    print(cost(X,y,final,lambdaa,p))
    np.savetxt('test.txt',final)

    return final2

def crossvalidation(theta):
    t=import_train_data()
    pc = t[1,:].size -2

    y = t[range(300,400),pc+1]
    #y = y[range(300,400),:]
    m = y.size
    y = y.reshape(m,1) #reshaping the original output in a matrix

    X = t[:,range(1,pc+1)]
    X = X[range(300,400),:]
    o = np.ones(m).reshape(m,1)
    X = np.append(o,X,axis = 1)

    return cost(X,y,theta)

def test_on_data(theta):
    t=import_test_data()    #Import test data
    pc = t[1,:].size -1     #Setting the parameter count

    ID = t[:,0]
    m = ID.size
    ID = ID.reshape(m,1)

    X = t[:,range(1,pc+1)]
    o = np.ones(m).reshape(m,1)
    X = np.append(o,X,axis = 1)

    np.savetxt('testres.txt',np.dot(X,theta))


theta=descent()
print(crossvalidation(theta))
test_on_data(theta)



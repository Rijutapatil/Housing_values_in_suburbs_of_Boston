import numpy as np
import math
import os

#Import Training Data
def import_train_data():
    file_dir = os.path.dirname(__file__)
    file_path= os.path.join(file_dir,'data/train.csv')
    train_data= np.loadtxt(file_path, dtype= float , delimiter= ',', skiprows= 1)
    return train_data

#Import Test data
def import_test_data():
    file_dir = os.path.dirname(__file__)
    file_path= os.path.join(file_dir,'data/test.csv')
    test_data= np.loadtxt(file_path, dtype= float , delimiter= ',', skiprows= 1)
    return test_data

#PNorm of theta
def norm(theta,p):
    return np.sum(np.absolute(theta) **p,axis=0)

#Cost function
def cost(X,y,theta,lambdaa,p):
    m = y.size
    J = (np.sum((np.dot(X,theta) - y)**2,axis=0) + (lambdaa*norm(theta,p) )) / (2*m)
    return J[0]

#Normal Equation
def NormalEquation(X, y, lambdaa):
    tbi = np.dot(np.transpose(X),X)
    tbi = tbi + lambdaa*np.identity(math.floor((tbi.size)**0.5))
    i = np.dot(np.linalg.inv(tbi),np.transpose(X))
    theta = np.dot(i,y)
    return theta

#Gradient Descent algorithm with penalisation on pnorm of theta
def gradientDescent(X,y,theta,alpha,iter,pnorm,lambdaa):
    m = y.size
    Jhistory = np.zeros(iter).reshape(iter,1)
    for i in range(0,iter):
        xtrans = np.transpose(X)
        loss = np.dot(X, theta) - y
        gradient = np.dot(xtrans, loss) / m
        theta = theta * (1 - (alpha * lambdaa * pnorm/ (2*m)))- (alpha * gradient)
        Jhistory[i,0] = cost(X,y,theta,lambdaa,pnorm)
    #return np.append(theta,Jhistory)
    return theta

#Outputting the results of the test data
def output(string,data):
    m=data.size
    id = np.array(list(range(m)))
    data = np.squeeze(np.asarray(data))
    outdat = np.column_stack((id.flatten(),data.flatten()))
    #final_data = np.append(['ID','MEDV'],final_data,axis=0)
    np.savetxt(string,outdat,delimiter=',',header="ID,MEDV")

#Normalizing a parameter vector
def normalize(parameter):
    mean = np.mean(parameter,dtype=float)
    variance = np.ptp(parameter)
    new_parameter = (parameter - mean)/variance
    return np.append([mean, variance],new_parameter)

#Main function
def descent():
    t=import_train_data() #Import training data
    pc = t[1,:].size -2     #Setting the parameter count

    y = t[range(0,300),pc+1]
    m = y.size
    y = y.reshape(m,1) #reshaping the original output in a matrix

    X = t[:,range(1,pc+1)]
    X = X[range(0,300),:]
    o = np.ones(m).reshape(m,1)
    X = np.append(o,X,axis = 1) #Appending column of 1s to the feature matrix

    theta = np.zeros(pc+1).reshape(pc+1,1) #Initializing theta vector to 0s

    iter = 400000   #setting the number of iterations
    alpha = 0.00000637   #setting the step size of descent
    lambdaa = 2     #setting the regularizer

    #Outputting files
    output('output.csv',test_on_data(gradientDescent(X,y,theta,alpha,iter,2,lambdaa)))
    output('output_p1.csv',test_on_data(gradientDescent(X,y,theta,alpha,iter,1.2,lambdaa)))
    output('output_p2.csv',test_on_data(gradientDescent(X,y,theta,alpha,iter,1.5,lambdaa)))
    output('output_p3.csv',test_on_data(gradientDescent(X,y,theta,alpha,iter,1.8,lambdaa)))

#Cross validate the model
def crossvalidation(thetax):
    t=import_train_data()
    pc = t[1,:].size -2

    y = t[range(300,400),pc+1]
    m = y.size
    y = y.reshape(m,1) #reshaping the original output in a matrix

    X = t[:,range(1,pc+1)]
    X = X[range(300,400),:]
    o = np.ones(m).reshape(m,1)
    X = np.append(o,X,axis = 1)

    return cost(X,y,thetax,1,2)

#Test on new Data set
def test_on_data(thetax):
    t=import_test_data()    #Import test data
    pc = t[1,:].size -1     #Setting the parameter count

    ID = t[:,0]
    m = ID.size
    ID = ID.reshape(m,1)

    X = t[:,range(1,pc+1)]
    o = np.ones(m).reshape(m,1)
    X = np.append(o,X,axis = 1)

    return np.dot(X,thetax)

descent()

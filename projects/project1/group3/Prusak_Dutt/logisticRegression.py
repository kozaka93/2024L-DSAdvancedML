import numpy as np
from enum import Enum

class Optimizer(Enum):
    SGD=1
    Adam=2
    IWLS=3
class LogisticRegression:
    def __init__(self, noOfIterations: int=1000, learningRate: float=0.001, optimizer: Optimizer=Optimizer.SGD, convError=0.00001, batchSize=64, printInfo=True):
        self.w= []
        self.noOfIterations = noOfIterations
        self.learningRate = learningRate
        self.costs=[]
        self.optimizer = optimizer
        self.convError = convError
        self.batchSize=batchSize
        self.printInfo=printInfo
        
    def fit(self,X,y):
        match self.optimizer:
            case Optimizer.SGD:
                return self.fitSgd(X,y)
            case Optimizer.Adam:
                return self.fitAdam(X,y)
            case Optimizer.IWLS:
                return self.fitIwls(X,y)
            
    def predict(self,X):
        predicted  = 1/(1+np.exp(-(np.dot(X,self.w).astype(float) +self.bias)).astype(float))
        predictedClasses = []
        for x in predicted:
            if x>0.5:
                predictedClasses.append(1)
            else:
                predictedClasses.append(0)

        return predictedClasses
    
    def fitSgd(self, X,y):
        # Initializing with weights 0 and 0 bias
        self.w = np.zeros(X.shape[1],dtype=np.float64)
        self.bias=np.float64(0)
        self.costs=[]
        for i in range(self.noOfIterations):
            p = np.random.permutation(len(X))
            shuffledX = X[p]
            shuffledY=y[p]

            for j in range(0,X.shape[0],self.batchSize):
                currentX = shuffledX[i:i+self.batchSize]
                currentY = shuffledY[i:i+self.batchSize]
                if(currentX.shape[0]<=0 or currentY.shape[0]<=0):
                    continue


                a = np.float64(np.dot(currentX,self.w) +self.bias)
                yHat=np.float64(1/(1+np.exp(-a)))


               # update weights
                weightChange =np.float64((1/currentX.shape[0])*np.dot(currentX.T,(yHat-currentY)))
                biasChange = np.float64((1/currentX.shape[0]) * np.sum(yHat-currentY))

                self.w = np.float64(self.w-self.learningRate*weightChange)
                self.bias=np.float64(self.bias-self.learningRate*biasChange)
            a = np.float64(np.dot(X,self.w) +self.bias)
            yHat=np.float64(1/(1+np.exp(-a)))
            yHat[yHat==1]=1-1e-08
            yHat[yHat==0]=1e-08
            self.costs.append((-1/X.shape[0])*(np.dot(y,np.log(yHat))+np.dot((1-y),np.log(1-yHat))))
            if i>0 and (np.abs(self.costs[-1]-self.costs[-2])<self.convError):
                if self.printInfo==True:
                    print("SGD Converged after "+str(i)+" iterations")
                break
        return self.costs
    def fitAdam(self,X,y):

        # 1. Init values
        # 2. Compute Gradients
        # 3. Get first and second moments
        # 4. Calculate bias corrections
        # 5. update weights
        self.w = np.zeros(X.shape[1])
        self.bias=0
        self.costs=[]
        # Values used by TensorFlow: beta1=0.9, beta2=0.999, epsilon=1e-08
        beta1=0.9
        beta2=0.999
        epsilon=1e-08
        moment1Weights=np.zeros(X.shape[1])
        moment2Weights=np.zeros(X.shape[1])
        moment1Bias=0
        moment2Bias=0
        
        for i in range(1,self.noOfIterations+1):
            
            a = np.dot(X,self.w) +self.bias
            a=a.astype(float)
            yHat=1.0/(1.0+np.exp(-a))

            # cost function (cross entropy loss)
            yHat[yHat==1]=1-1e-08
            yHat[yHat==0]=1e-08
            self.costs.append((-1.0/X.shape[0])*(np.dot(y,np.log(yHat))+np.dot((1.0-y),np.log(1.0-yHat))))
            
            gradientWithRespectToWeights=np.dot(X.T,yHat-y)
            gradientWithRespectToBias = np.sum(yHat-y)
            moment1Weights=beta1*moment1Weights+(1-beta1)*gradientWithRespectToWeights
            moment2Weights=beta2*moment2Weights+(1-beta2)*gradientWithRespectToWeights*gradientWithRespectToWeights
            mhatWeights = moment1Weights/(1.0-beta1**i)
            vhatWeights = moment2Weights/(1.0-beta2**i)

            self.w = self.w-self.learningRate*mhatWeights/(np.sqrt(vhatWeights)+epsilon)

            moment1Bias=beta1*moment1Bias+(1.0-beta1)*gradientWithRespectToBias
            moment2Bias=np.dot(beta2*moment2Bias+(1.0-beta2),gradientWithRespectToBias*gradientWithRespectToBias)

            mhatBias = float(moment1Bias)/(1.0-float(beta1**i))
            vhatBias = float(moment2Bias)/(1.0-float(beta2**i))

            self.bias = self.bias-self.learningRate*mhatBias/(np.sqrt(vhatBias)+epsilon)
            if i>1 and (np.abs(self.costs[-1]-self.costs[-2])<self.convError):
                if self.printInfo==True:
                    print("Adam Converged after "+str(i)+" iterations")
                break
        return self.costs
    def fitIwls(self,X,y):
        self.w = np.zeros(X.shape[1])
        self.costs=[]
        self.bias=0
        for i in range(1,self.noOfIterations+1):
            a = np.dot(X,self.w) 
            a=a.astype(float)
            yHat=1.0/(1.0+np.exp(-a))

            self.costs.append((-1.0/X.shape[0])*(np.dot(y,np.log(yHat,out=np.zeros_like(yHat), where=(yHat!=0)))+np.dot((1.0-y),np.log(1.0-yHat,out=np.zeros_like(yHat), where=(yHat!=1)))))
            # using formula from lecture slides (X^TWX )^−1X^TWz
            # z = X βold + W^−1(y − p), Beta being weights
            pp=yHat*(1-yHat)
            W = np.diag(pp)
            z = a+ np.dot(np.linalg.pinv(W),y-yHat)
            # (np.divide((y-yHat),pp,out=np.zeros_like(pp),where=pp!=0))
            self.w=np.dot(np.dot(np.dot(np.linalg.pinv(np.dot(np.dot(X.T,W),X)),X.T),W),z)
            if i>1 and (np.abs(self.costs[-1]-self.costs[-2])<self.convError):
                if self.printInfo==True:
                    print("IWLS Converged after "+str(i)+" iterations")
                break
        return self.costs
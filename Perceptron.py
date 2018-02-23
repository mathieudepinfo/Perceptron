import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time as time

def activation(x):
    
    return np.tanh(x)

def dactivation(x):
    
    return 1.0-np.tanh(x)**2
    
class perceptron:
    
    def __init__(self,*args):
        
        self.forme=args
        self.nbcouches=len(args)
        self.couches=[]
        self.couches.append(np.ones(args[0]+1))
        for i in range(1,len(args)):
            self.couches.append(np.ones(args[i]))
            
        self.poids=[]
        self.deltapoids=[]
        
        for i in range(len(args)-1):
            self.poids.append(np.random.random((self.couches[i].size,self.couches[i+1].size)))
            self.deltapoids.append(np.zeros((self.couches[i].size,self.couches[i+1].size)))
        
    def propagation(self,data):
        
        self.couches[0][0:-1]=data
        
        for i in range(1,self.nbcouches):
            
            self.couches[i]=activation(np.dot(self.couches[i-1],self.poids[i-1]))
            
        
        return self.couches[self.nbcouches-1]
        
    def retropropagation(self,data,vitesse=0.1,inertie=0.1):
        
        err=[]
        for i in range(self.nbcouches):
            err.append(np.ones(self.couches[i].size))
        
        err[self.nbcouches-1]=(data-self.couches[-1])*dactivation(self.couches[-1])
        
        for k in range(self.nbcouches-2,0,-1):
            
            for n in range(self.couches[k].size):
                
                err[k][n]=dactivation(self.couches[k][n])*np.dot(np.transpose(self.poids[k][n]),err[k+1])
                
        for k in range(1,self.nbcouches):
            for i in range(self.couches[k-1].size):
                for j in range(self.couches[k].size):
                    
                    self.poids[k-1][i][j]=self.poids[k-1][i][j]+vitesse*self.couches[k-1][i]*err[k][j]+inertie*self.deltapoids[k-1][i][j]
                    self.deltapoids[k-1][i][j]=self.couches[k-1][i]*err[k][j]
        
        return (err[len(err)-1]**2).sum()
        
def apprendre(data,perceptron,n):
    
    
        
    for i in range(n):
        
        perceptron.propagation(data[i%len(data)][0])
        perceptron.retropropagation(data[i%len(data)][1])





    

t0 = time.clock()
samples = np.zeros(500, dtype=[('x',  float, 1), ('y', float, 1)])
samples['x'] = np.linspace(0,1,500)
samples['y'] = np.sin(samples['x']*np.pi)


network=perceptron(1,10,1)
for i in range(100000):
    n = np.random.randint(samples.size)
        
    network.propagation(np.array([samples['x'][n]]))
    network.retropropagation(np.array([samples['y'][n]]))

plt.figure(figsize=(10,5))
# Draw real function
x,y = samples['x'],samples['y']
plt.plot(x,y,color='b',lw=1)
# Draw network approximated function

for i in range(samples.shape[0]):
    y[i] = network.propagation(x[i])
plt.plot(x,y,color=(1,0,0),lw=3)

print(time.clock()-t0)
plt.axis([0,1,0,1])
plt.show()




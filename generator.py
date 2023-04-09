import matplotlib.pyplot as plt
import numpy as np
import time
from itertools import product
from numba import jit

def UniformDistribution(n,seed=int(time.time())%100000):
    #linear congruential generator
    m=244944
    a=1597
    c=51749
    for i in range(n):
        seed=(a*seed+c)%m
        #extremely low probability that seed=0, this should be avoided
        if seed==0:
            seed=1
        yield seed/m
        
def ExponentialDistribution(n,seed=int(time.time())%100000):
    #inverse transform method of exponential distribution
    uniform_list=list(UniformDistribution(n,seed))
    for i in range(n):
        yield -np.log(1-uniform_list[i])

def InverseNormalCdf(p):
    #abramowitz and stegun method
    if p<0 or p>1:
        raise ValueError('p must be between 0 and 1')
    if p>0.5:
        return -InverseNormalCdf(1-p)
    c0=np.float64(2.515517)
    c1=np.float64(0.802853)
    c2=np.float64(0.010328)
    d1=np.float64(1.432788)
    d2=np.float64(0.189269)
    d3=np.float64(0.001308)
    t=(np.log(1/(p**2)))**0.5
    xp=t-(c0+c1*t+c2*t**2)/(1+d1*t+d2*t**2+d3*t**3)
    return -xp
    
def NormalDistribution_inverse_transform(n,seed=int(time.time())%100000):
    #inverse transform method of normal distribution
    uniform_list=list(UniformDistribution(n,seed))
    return [InverseNormalCdf(i) for i in uniform_list]
        
def DoubleExpoential(x):
    #the inverse of the cdf of the double exponential distribution or laplace distribution
    if x<0.5:
        return np.log(x*2)
    else:
        return -np.log(2-(x*2))
        
def NormalDistribution_accept_reject(n,seed=int(time.time())%100000):
    #the g(x) we select is 
    #g(x)=exp(x)/2 x<=0
    #g(x)=exp(-x)/2 x>0
    normal_list=[]
    #c is the parameter of the acceptance-rejection method
    #1/c denotes the probability that a point is accpeted
    #approximately c is 1.315489246958914
    c=(2/np.pi)**0.5*np.exp(0.5)
    while(len(normal_list)<n):
        uniform_list=list(UniformDistribution(n+1,seed=seed+len(normal_list)))
        for i in range(n):
            x=DoubleExpoential(uniform_list[i])
            gx=0.5*np.exp(-x) if x>0 else 0.5*np.exp(x)
            fx=1/np.sqrt(2*np.pi)*np.exp(-x**2/2)
            if(uniform_list[i+1]<fx/(gx*c)):
                normal_list.append(x)
    return normal_list[:n]

def NomalDistribution_box_muller(n,seed=int(time.time())%100000):
    #box muller method
    uniform_list=list(UniformDistribution(2*n,seed))
    normal_list=[]
    for i in range(n):
        y1=(-2*np.log(uniform_list[2*i]))**0.5*np.cos(2*np.pi*uniform_list[2*i+1])
        y2=(-2*np.log(uniform_list[2*i]))**0.5*np.sin(2*np.pi*uniform_list[2*i+1])
        normal_list.append(y1)
        normal_list.append(y2)
    return normal_list[:n]
            
        
def CholeskyDecomposition(A):
    #cholesky decomposition
    #it seems quite elegant in python:)
    n=A.shape[0]
    L=np.zeros((n,n))
    for i in range(n):
        for j in range(i+1):
            if i==j:
                L[i,j]=(A[i,j]-np.sum(L[i,:j]**2))**0.5
            else:
                L[i,j]=(A[i,j]-np.sum(L[i,:j]*L[j,:j]))/L[j,j]
    return L,L.T
    
def NormalCdf(x):
    b1=np.float64(0.31938153)
    b2=np.float64(-0.356563782)
    b3=np.float64(1.781477937)
    b4=np.float64(-1.821255978)
    b5=np.float64(1.330274429)
    p=np.float64(0.2316419)
    c=np.log(2*np.pi)/2
    a=np.abs(x)
    t=1/(1+a*p)
    s=((((b5*t+b4)*t+b3)*t+b2)*t+b1)*t
    y=s*np.exp(-a**2/2-c)
    if x>0:
        y=1-y
    return y

def HaltonTransform(a,n):
    #the sequence is inverse of base-n representation of a
    #this is for quasi random number generation, but not used in this project so far
    def base_n(a,n):
        place_list=[]
        while a>=n:
            k=a%n
            a=(a-k)/n
            place_list.append(k)
        place_list.append(a)
        return place_list

    sequence=base_n(a,n)
    for i,j in enumerate(sequence):
        sequence[i]=sequence[i]/n**(i+1)
    return sum(sequence)
    

def shuffle(x,seed=int(time.time())%100000):
    #shuffle the list x
    random_list=list(UniformDistribution(len(x),seed))
    for i in reversed(range(1,len(x))):
            j = int(random_list[i] * (i + 1))
            x[i], x[j] = x[j], x[i]
    return x

def NormalDistribution_stratified_inversed(n,stratum=10,seed=int(time.time())%100000):
    #default stratum number is 10
    #actually stratified sampling can be used for any distribution generation for variance reduction
    uniform_list=list(UniformDistribution(n,seed))
    stratum_quota=n/stratum
    for i in range(n):
        uniform_list[i]=float(i//stratum_quota)/stratum+uniform_list[i]/stratum
    normal_list=[InverseNormalCdf(i) for i in uniform_list]
    normal_list=shuffle(normal_list)
    return normal_list



# def NormalDistribution_stratified_accept_reject(n,stratum=10,seed=int(time.time())%100000):
#     #the g(x) we select is 
#     #g(x)=exp(x)/2 x<=0
#     #g(x)=exp(-x)/2 x>0
#     normal_list=[]
#     #c is the parameter of the acceptance-rejection method
#     #1/c denotes the probability that a point is accpeted
#     #approximately c is 1.315489246958914
#     c=(2/np.pi)**0.5*np.exp(0.5)
#     while(len(normal_list)<n):
#         uniform_list=list(UniformDistribution(n+1,seed=seed+len(normal_list)))
#         for i in range(n):
#             x=DoubleExpoential(uniform_list[i])
#             gx=0.5*np.exp(-x) if x>0 else 0.5*np.exp(x)
#             fx=1/np.sqrt(2*np.pi)*np.exp(-x**2/2)
#             if(uniform_list[i+1]<fx/(gx*c)):
#                 normal_list.append(x)
    
#     uniform_list=[NormalCdf(i) for i in normal_list]
#     stratum_quota=n/stratum
#     for i in range(n):
#         uniform_list[i]=float(i//stratum_quota)/stratum+uniform_list[i]/stratum     
#     normal_list=[InverseNormalCdf(i) for i in uniform_list]
#     normal_list=shuffle(normal_list)
#     return normal_list[:n]

def stratify(x,i,stratum):
    return float(i/stratum)+x/stratum

@jit
def NormalDistribution_stratified_accept_reject(n,v,stratum=4,seed=int(time.time())%100000):
    #every random variable has n samples(totally v varibales) and groups in the variable is stratum
    #return normal list and the number of groups
    normal_list=NormalDistribution_accept_reject(n*v)
    uniform_list=[NormalCdf(i) for i in normal_list]
    iter_list=list(product(*[range(stratum) for _ in range(v)]))
    def stratify(x,i,stratum):
        return float(i/stratum)+x/stratum
    subgroup_size=n*v/len(iter_list)
    for i in range(len(iter_list)):
        group_info=iter_list[i]
        for j in range(int(subgroup_size/v)):
            for k in range(v):
                uniform_list[int(k+j*v+subgroup_size*i)]=stratify(uniform_list[int(k+j*v+subgroup_size*i)],group_info[k],stratum)
    normal_list=[InverseNormalCdf(i) for i in uniform_list]
    return normal_list,len(iter_list)


if __name__=='__main__':
    result=NormalDistribution_stratified_accept_reject(1000,4)
    print(result)
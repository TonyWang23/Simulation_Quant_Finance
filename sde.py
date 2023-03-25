import generator
import numpy as np

#The pricing method below is only used for this example to ilustrate the technique
#More standard and integrated pricing method should be used in practice

def PricingResultGBM(r,sigma,S0,T,Z):
    #Europe call option pricing result
    S=S0*np.exp((r-0.5*sigma**2)*T+sigma*(T**0.5)*Z)
    return S

def OptionPricingResultGBM(r,sigma_list,S0_list,T,n,cov_matrix,K):
    #most naive method with inverse transform method, used as a benchmark
    sde_num=len(sigma_list)
    random_list=generator.NormalDistribution_inverse_transform(sde_num*n)
    L,_=generator.CholeskyDecomposition(cov_matrix)
    H=[]
    for i in range(n):
        Z=np.dot(L,random_list[i*sde_num:(i+1)*sde_num])
        s_result=[PricingResultGBM(r,sigma_list[j],S0_list[j],T,Z[j]) for j in range(sde_num)]
        H.append( np.exp(-r*T)*np.max([np.max(s_result)-K,0]))
    return H

def OptionPricingResultGBM_antithetic(r,sigma_list,S0_list,T,n,cov_matrix,K):
    #athitethic sampling method is used for variance reduction
    sde_num=len(sigma_list)
    random_list=generator.NormalDistribution_accept_reject(int(sde_num*n))
    L,_=generator.CholeskyDecomposition(cov_matrix)
    H=[]
    for i in range(int(n)):
        Z=np.dot(L,random_list[i*sde_num:(i+1)*sde_num])
        s_result_1=[PricingResultGBM(r,sigma_list[j],S0_list[j],T,Z[j]) for j in range(sde_num)]
        s_result_2=[PricingResultGBM(r,sigma_list[j],S0_list[j],T,-Z[j]) for j in range(sde_num)]
        c_result=(np.exp(-r*T)*np.max([np.max(s_result_1)-K,0])+np.exp(-r*T)*np.max([np.max(s_result_2)-K,0]))/2
        H.append(c_result)
    return H

def OptionPricingResultGBM_stratified(r,sigma_list,S0_list,T,n,cov_matrix,K):
    #only used stratifeid sampling method
    sde_num=len(sigma_list)
    random_list=generator.NormalDistribution_stratified_inversed(sde_num*n)
    L,_=generator.CholeskyDecomposition(cov_matrix)
    H=[]
    for i in range(n):
        Z=np.dot(L,random_list[i*sde_num:(i+1)*sde_num])
        s_result=[PricingResultGBM(r,sigma_list[j],S0_list[j],T,Z[j]) for j in range(sde_num)]
        H.append( np.exp(-r*T)*np.max([np.max(s_result)-K,0]))
    return H

def OptionPricingResultGBM_antithetic_stratified(r,sigma_list,S0_list,T,n,cov_matrix,K):
    #both stratified sampling and antithetic sampling are used
    sde_num=len(sigma_list)
    random_list=generator.NormalDistribution_stratified_inversed(int(sde_num*n))
    L,_=generator.CholeskyDecomposition(cov_matrix)
    H=[]
    for i in range(int(n)):
        Z=np.dot(L,random_list[i*sde_num:(i+1)*sde_num])
        s_result_1=[PricingResultGBM(r,sigma_list[j],S0_list[j],T,Z[j]) for j in range(sde_num)]
        s_result_2=[PricingResultGBM(r,sigma_list[j],S0_list[j],T,-Z[j]) for j in range(sde_num)]
        c_result=(np.exp(-r*T)*np.max([np.max(s_result_1)-K,0])+np.exp(-r*T)*np.max([np.max(s_result_2)-K,0]))/2
        H.append(c_result)
    return H


if __name__=='__main__':
    import matplotlib.pyplot as plt
    r=0.02
    sigma_list=[0.1,0.1,0.1,0.2]
    S0_list=[45,50,45,55]
    T=0.5
    n=10000
    cov_matrix=np.array([[1,0.3,-0.2,0.4],[0.3,1,-0.3,0.1],[-0.2,-0.3,1,0.5],[0.4,0.1,0.5,1]])
    K=55
    H=OptionPricingResultGBM(r,sigma_list,S0_list,T,n,cov_matrix,K)
    print(np.mean(H))

    plt.hist(H,bins=20)
    plt.xlim(0,5)
    plt.show()

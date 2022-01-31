"""
A HVUS tool for unbiasedly estimating HVUS and its variance based on a fast 
version algorithm, comparing two classifiers' performance, and visualizing 
the three-class ROC surface.

"""

# Created on Sun Jul 19 14:33:39 2020
#
# Authors: LIU shun <sliu5222@foxmail.com>
#          YANG Junjie <Junjie.Yang@l2s.centralesupelec.fr>
#       
# License: MIT license




import numpy as np
import itertools 
import scipy.stats as ss
import os

    



def _con_general_counter_matrix(label,data): 
    """
        Constuct counter matrix CM for computing HVUS based on dynamic programming

    """
    
    indexSort = np.argsort(data)
    dataSort = np.sort(data)
    labelSort = label[indexSort]
    class_num = len(set(labelSort))
    CM = np.zeros((class_num,len(labelSort)))
    for i in range(len(labelSort)):
        CM[int(labelSort[i])-1][i] = 1

    return CM

def _DP_continuous_eventcount(CM, update_rule):
    """
        Dynamic programming algorithm for attaining HVUS in terms of counter matrix

    """
    
    
    height, width = CM.shape
    lur = len(update_rule)
    L = CM[0][0:width-lur]
    current_end_index = lur
    for i in range(1,height):
        current_end_index = current_end_index - 1
        ud_elements = CM[i][i:width-current_end_index]
        L = np.cumsum(L)*ud_elements
        
    L = np.cumsum(L)
    s = L[-1]
    return s

def _bitget(byteval, idx):
    """
        A binary code for getting index of Z2(Z1)

    """

    return ((byteval & (1 << idx)) != 0)


def _unoredr2order(label,data):
    """
        Turn unorder data to order data according to their mean value 

    """
    
    labelSet = list(set(label))
    dataMean = np.array([ np.mean(data[label==i]) for i in labelSet ])
    labelSetSort = np.arange(1,len(labelSet)+1)[np.argsort(dataMean)] 
    label2 = label.copy()
    for i,l in enumerate(labelSet):
        label2[label==l] = labelSetSort[i]
    data2 = data[np.argsort(label2)]
    label2.sort()
    return label2,data2

def order_hvus(label,data): 
    """
        Compute ordered hvus

    Parameters
    ----------
    label : array-like of shape (n_samples,)
        samples' label indicating whihch class a sample belongs to 
        
    data : array-like of shape (n_samples,)
        samples' value

    Returns
    -------
    hvus : float64
        hvus value 

    """
    

    label,data = _unoredr2order(label,data)
    #label = np.array([int(i) for i in label])

    CM = _con_general_counter_matrix(label,data)
    class_num = len(set(label))
    subclass_num = np.array([(label==i).sum() for i in set(label)])
    update_rule = 'B' * (class_num-1)
    
    hvus = _DP_continuous_eventcount(CM, update_rule)/subclass_num.prod()

    return hvus

def hvus(label,data):
    """
        Estimate hvus and its variance based on graph theory 
        

    Parameters
    ----------
    label : array-like of shape (n_samples,)
        samples' label indicating whihch class a sample belongs to 
        
    data : array-like of shape (n_samples,)
        samples' value

    Returns
    -------
    hvus : float64
        hvus value 
    var_hvus : float64
        the variance of hvus

    """
    

    

    hv = order_hvus(label,data)
    
    label,data = _unoredr2order(label,data)
    label = np.array([int(i) for i in label])

    data = data[label.argsort()]
    label.sort()


    var_hvus = 0
    class_num = len(set(label))

    gamaIndex = np.zeros((2**class_num,class_num))

    for c1 in range(2**class_num):
        for c2 in range(class_num):
            gamaIndex[c1][c2] = _bitget(c1,c2)
        

    data = np.r_[-np.inf,data,np.inf]
    label = np.r_[0,label,class_num+1]+1
    
    
    n = len(data)

    r = ss.rankdata(data)

    si = data.argsort()


    label = label[si]
    subclass_num = np.array([np.sum((label==i)) for i in set(label)])

    gamaIndex = np.insert(gamaIndex,0,1,axis=1)
    gamaIndex = np.insert(gamaIndex,class_num+1,1,axis=1)

    c = np.cumsum(subclass_num)
    c = np.append(0,c[:])

    uic = np.zeros((2**class_num))

    for t in range(2**class_num):
        current_divide = gamaIndex[t][:]
        V = [i for i in range(len(current_divide)) if current_divide[i]==1]


        ww = [[]] * (len(V) - 1)


        for p in range(len(V)-1):
            k = V[p]
            l = V[p+1]

            ww[p] = np.zeros((subclass_num[k],subclass_num[l]))


            if l == k+1:
                for i in range(subclass_num[k]):
                    for j in range(subclass_num[l]):
                        
                        if data[c[k]+i] < data[c[l]+j]:
                            ww[p][i][j] = 1
                        # print(k,l,i,j,p)
            else:
                for i in range(subclass_num[k]):
                    w = np.zeros(l-k-1)
                    for j in range(int(r[c[k]+i]),n):
                        m = label[j] - 1

                        if m == (k+1):
                            w[0]  = w[0] + 1
                        
                        if (k+1 < m) and (m < l):
                            w[m-k-1] += w[m-k-2]
                        if m == l:
                            #print(si[j]-c[m])
                            ww[p][i][si[j]-c[m]] = w[l-k-2] 
                        
                        
        gama = [[]] * len(V)
        st = 0
        gama[st].append(1)
    
        for k in range(len(V)-1):
            k1 = V[k]
            k2 = V[k+1]
            st += 1
            gama[st] = np.zeros(subclass_num[k2])
            for a in range(subclass_num[k1]):
                for b in range(subclass_num[k2]):
                    gama[st][b] += ww[st-1][a][b] * ww[st-1][a][b] * gama[st-1][a]
        uic[t] = gama[-1]

    ic = uic

    gamacopy = gamaIndex.copy()

    for t in range(1,2**class_num-1):
        oi = gamacopy[t][:]
        zi = (gamacopy[t][:] == 0)

        zc = int(sum(zi))

    
    
        inter = np.zeros((2**zc,zc))
        for c1 in range(2**zc):
            for c2 in range(zc):
                inter[c1][c2] = _bitget(c1,c2)
  
    
        for i in range(1,len(inter)):
            oi[zi] = inter[i][:]
            current_block = oi[1:-1]


            block_eq_ones = np.array([l for l in range(len(current_block)) if current_block[l]==1])
            ic[t] += (-1) ** sum(inter[i][:]) * uic[sum(2**block_eq_ones)] 
        

    for t in range(1,2**class_num):
        if t == (2**class_num-1):
            var_hvus += (ic[t]/ (subclass_num.prod()) - hv*hv)
        
        else:
            temp_prod = (subclass_num[gamaIndex[t][:]==0]-1).prod()
            var_hvus += temp_prod*(ic[t]/(subclass_num.prod()*temp_prod)-hv**2)
                
    var_hvus = var_hvus/((subclass_num[1:-1]-1).prod())

    return hv,var_hvus




def cov_hvus(data1,data2,label):
    """
        Estimate the covariance of hvus between two classifiers/features based on graph theory

    Parameters
    ----------
    data1 : array-like of shape (n_samples,)
        samples' value from classifier1
    data2 : array-like of shape (n_samples,)
        samples' value from classifier1
    label : array-like of shape (n_samples,)
        samples' label indicating whihch class a sample belongs to 

    Returns
    -------
    cov_hvus : float64
        the covariance of hvus

    """
    

    cov_hv = 0
    hvus2 = order_hvus(label,data2)
    hvus1 = order_hvus(label,data1)


    subclass_num1 = np.array([(label==i).sum() for i in set(label)])
    subclass_num2 = subclass_num1


    r = len(subclass_num1)
    appdata1c = np.append(0,subclass_num1)
    cs = np.cumsum(appdata1c)

    gamaIndex = np.zeros((2**r,r))

    for c1 in range(2**r):
        for c2 in range(r):
            gamaIndex[c1][c2] = _bitget(c1,c2)
        
    V = [[]]* (r+1)
    Vi = [[]] * r
    Ni = [[]] * r

    for t in range(1,2**r):
        for k in range(r):
            if gamaIndex[t][k] == 1:
                V[k] = np.zeros((1,subclass_num1[k]))
                Vi[k] = np.array((range(subclass_num1[k]),range(subclass_num1[k])))
                Ni[k] = np.array((np.ones(subclass_num1[k]),range(subclass_num1[k])))
            else:
                V[k]= np.zeros((subclass_num1[k],subclass_num1[k]))
                Vi[k] = np.zeros((2,subclass_num1[k]**2))
                Ni[k] = np.zeros((2,subclass_num1[k]**2))
                length = 0
        
                for i in range(subclass_num1[k]):
                    for j in range(subclass_num1[k]):

                        Vi[k][0,length] = i
                        Vi[k][1,length] = j
                        Ni[k][0,length] = i
                        Ni[k][1,length] = j
                        length += 1
                    
        V[0] = np.ones(V[0].shape)
    
        if gamaIndex[t][0] != 1:
            for i in range(V[0].shape[0]):
        
                V[0][i][i] = 0
            

    
        for k in range(r-1):
            for va in range(V[k].size):   
                _,Vk_width = V[k].shape
                Vk_row_index = va // Vk_width
                Vk_col_index = va % Vk_width
                for vb in range(V[k+1].size):
                    _,Vkpone_width =V[k+1].shape
                    Vkpo_row_index = vb//Vkpone_width
                    Vkpo_col_index = vb%Vkpone_width
                    if gamaIndex[t][k+1] != 1 and (Vi[k+1][0][vb]==Vi[k+1][1][vb]):
                        V[k+1][Vkpo_row_index][Vkpo_col_index] = 0
                    
                    else:
                        if data1[int(cs[k]+Vi[k][0][va])]<data1[int(cs[k+1]+Vi[k+1][0][vb])] and \
                            data2[int(cs[k]+Vi[k][1][va])]<data2[int(cs[k+1]+Vi[k+1][1][vb])]:
                            V[k+1][Vkpo_row_index][Vkpo_col_index] += V[k][Vk_row_index,Vk_col_index]
                        
                        
        V[r] = sum(sum(V[r-1]))
    
        temp_prod = (subclass_num1[gamaIndex[t][:]==0]-1).prod()
        
    
        cov_hv += temp_prod * (V[r] / (subclass_num1.prod()*temp_prod) - hvus1*hvus2)
        
    cov_hv = cov_hv / subclass_num1.prod()
                    

    return cov_hv

def z_hypothetical_test(data1,data2,label,alpha=0.05):
    """
        Z test for comparing performance of two classifers/features with significant level alpha.

    Parameters
    ----------
    data1 : array-like of shape (n_samples,)
        samples' value from classifier1
    data2 : array-like of shape (n_samples,)
        samples' value from classifier1
    label : array-like of shape (n_samples,)
        samples' label indicating whihch class a sample belongs to 
    alpha : float64, default=0.05
        significant level

    Returns
    -------
    z : float64
        z statistics
    threshold : float64
        a threshold under the significant level a for determing if the two classifers have
        significant difference
        
        if z > threshold:
            Classifier2 is better than classifer1
        if z < -threshold:
            Classfier1 is better than classifer2
        otherwise:
            These two Classifiers do not have a significant difference
    """
    
    threshold = ss.norm.ppf(1-alpha/2)
    
    hvus1,var_hvus1 = hvus(label,data1)
    hvus2,var_hvus2 = hvus(label,data2)
    
    cov_1_2 = cov_hvus(data1,data2,label)     
           
    
    z = (hvus2 - hvus1) / np.sqrt((var_hvus1 + var_hvus2 - 2 * cov_1_2))
    
    return z,threshold



def plot_vus(label,data):
    """
    

    Parameters
    ----------
    label : array-like of shape (n_samples,)
        samples' label indicating whihch class a sample belongs to 
    data : array-like of shape (n_samples,)
        samples' value

    Returns
    -------
    P1,P2,P3 :
        3D data of VUS surface
        

    """
    
    label,data = _unoredr2order(label,data)
    labelSet = list(set(label))
    labelSet.sort()
    
    data1 = data[label==labelSet[0]]
    data2 = data[label==labelSet[1]]
    data3 = data[label==labelSet[2]]

    dataSet = np.array(list(set(data)))
    ths = np.r_[-np.inf,np.sort(data)]
    subclass_num = np.array([(label==i).sum() for i in labelSet])
    P1 = np.zeros((len(ths),len(ths)))
    P2 = np.zeros((len(ths),len(ths)))
    P3 = np.zeros((len(ths),len(ths)))
    for i,th1 in enumerate(ths):
        for j,th2 in enumerate(ths):
            P1[i,j] = (data1<= th1).mean()
            P3[i,j] = ( (data2 > th1) & (data2 <= th2) ).mean()
            P2[i,j] = (data3> th2).mean()
    return P1,P2,P3

def plot_roc(label,data):
    """
    

    Parameters
    ----------
    label : array-like of shape (n_samples,)
        samples' label indicating whihch class a sample belongs to 
    data : array-like of shape (n_samples,)
        samples' value

    Returns
    -------
    fpr,tpr :
        2D data for ROC
        

    """
    label,data = _unoredr2order(label,data)
    labelSet = np.unique(label)
    labelSet.sort()    
    data1 = data[label==labelSet[0]]
    data2 = data[label==labelSet[1]]
    ths = np.r_[-np.inf,np.sort(data)]
    fpr = np.array([(data1>=t).mean() for t in ths])
    tpr = np.array([(data2>=t).mean() for t in ths])

    return fpr,tpr


    
    
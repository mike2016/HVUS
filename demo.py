"""
    HVUS tool demo

"""

# Created on Tue Oct 27 10:19:05 2020
# Authors: LIU shun <sliu5222@foxmail.com>
#          YANG Junjie <Junjie.Yang@l2s.centralesupelec.fr>
#       
# License: MIT license



from HVUS import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


score1 = np.random.randn(100)+1
score2 = np.random.randn(100)+2
score3 = np.random.randn(100)+3
label1 = 1*np.ones(100)
label2 = 2*np.ones(100)
label3 = 3*np.ones(100)
score = np.r_[score1,score2,score3]
label = np.r_[label1,label2,label3]

## example 1
hv,var_hv = hvus(label,score)
print('HVUS: {}, Var: {}'.format(hv,var_hv))


## example 2
P1,P2,P3 = plot_vus(label,score)
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(P1,P2,P3)
plt.show()

## example 3
score1 = np.random.randn(100)+1
score2 = np.random.randn(100)+2
label1 = 1*np.ones(100)
label2 = 2*np.ones(100)
score = np.r_[score1,score2]
label = np.r_[label1,label2]
fpr,tpr = plot_roc(label,score)
plt.plot(fpr,tpr)
plt.show()


##example 4
scoreA1 = np.random.randn(20)+1
scoreA2 = np.random.randn(20)+2
scoreA3 = np.random.randn(20)+3
scoreA = np.r_[scoreA1,scoreA2,scoreA3]

scoreB1 = np.random.randn(20)+3
scoreB2 = np.random.randn(20)+1
scoreB3 = np.random.randn(20)+2
scoreB = np.r_[scoreB1,scoreB2,scoreB3]

label1 = 1*np.ones(20)
label2 = 2*np.ones(20)
label3 = 3*np.ones(20)
label = np.r_[label1,label2,label3]

z,th = z_hypothetical_test(scoreA,scoreB,label,alpha=0.05)
print('z statistic: {}; threshold" {}'.format(z,th))





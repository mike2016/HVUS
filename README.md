The hyper ROC surface and the corresponding hypervolume under the surface (HVUS) is a metric for assessing the classifying capabilities of a multi-classifier.  Here we develop the HVUS tool for unbiasedly estimating HVUS and its variance based on a fast version algorithm, comparing two classifiers' performance, and visualizing the three-class ROC surface.

## Installation

---

- Clone this repository 
```
git clone https://github.com/mike2016/HVUS.git
```

- Go inside the package
```
cd HVUS
```


- Install package
```
python setup.py install
```

#### Dependencies

- numpy
- scipy



## Usage
---
We take a three-class problem as an example to show how to use the HUVS tool to calculate HUVS and its variance, compare two classifier's performance, and visualize three-class ROC surface. Once we finished the HVUS tool installation, we can import the HVUS tool directly in python environment. 

### Calculate HVUS and variance
First, let's create a random vector to simulate score values obtained from a three-class classifier, where *score1*, *socre2* and *score3*  with different mean indicate the score of three classes, respectively. We also create a vector named *label* to represent samples' true class label. Then we call the function **HVUS​** to estimate HVUS value and its variance.

```python
from HVUS import *
import numpy as np

score1 = np.random.randn(100)+1
score2 = np.random.randn(100)+2
score3 = np.random.randn(100)+3
label1 = 1*np.ones(100)
label2 = 2*np.ones(100)
label3 = 3*np.ones(100)

score = np.r_[score1,score2,score3]
label = np.r_[label1,label2,label3]

hvus_value,var_hvus = hvus(label,score)
print('HVUS: {}, Var: {}'.format(hvus_value,var_hvus))
```
### Plot three-class ROC surface
Based on the simulated data above, we can visualize the three-class ROC surface by calling the function **plot_vus​**. Note that this function only returns 3D data of the surface. You can plot the surface easily by using the matplotlib package. 

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

P1,P2,P3 = plot_vus(label,score)
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(P1,P2,P3)
plt.show()
```



### Compare two classifiers
Here, we generate two random vectors *scoreA* and *scoreB* representing the score results of classifierA and classifierB. In these two classifiers, the means of score for three classes are different, while the corresponding label of samples is the same, denoted as *label​*. Now we use the function **z_hypothetical_test** to calculate the Z statistics and its threshold.

```python
from HVUS import *
import numpy as np

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
```

If *z>th*, classifierB is better than classiferA; If *z<-th*, classfierA is better than classiferB; Otherwise, these two classifiers do not have a significant difference.


## References
[1]: Liu, S., Zhu, H., Yi, K., Sun, X., Xu, W., and Wang, C. (2020). Fast and unbiased estimation of volume under ordered three-class roc surface (vus) with continuous or discrete measurements. IEEE Access, 8, 136206–136222.

[2]: Nakas, C. T. and Yiannoutsos, C. T. (2004). Ordered multiple-class roc analysis with continuous measurements. Statistics in medicine, 23(22), 3437–3449.

[3]: Waegeman, W., De Baets, B., and Boullart, L. (2008). On the scalability of ordered multi-class roc analysis. Computational Statistics & Data Analysis, 52(7), 3371– 3388.

[4]: Li, Jialiang, and Jason P. Fine. "ROC analysis with multiple classes and multiple tests: methodology and its application in microarray studies." Biostatistics 9.3 (2008): 566-576.

[5]: Novoselova, Natalia, et al. "HUM calculator and HUM package for R: easy-to-use software tools for multicategory receiver operating characteristic analysis." Bioinformatics 30.11 (2014): 1635-1636.

[6]: Hanley, James A., and Barbara J. McNeil. "A method of comparing the areas under receiver operating characteristic curves derived from the same cases." Radiology 148.3 (1983): 839-843.



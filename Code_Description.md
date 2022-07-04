# Interpolation with GPR method
This program is used to perform interpolation with Gaussian process regression method to predict the target values at given points. 
## Environment
python 3.9.1

## Program
    import matplotlib.pyplot as plt 
    import numpy as np
    from matplotlib.ticker import AutoMinorLocator 
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    from sklearn.gaussian_process.kernels import ConstantKernel as C 
* RBF: the SE kernel in Eq.(20) with $s_f^2=1$;
* Whitekernel: the noise term in Eq.(23);
* ConstantKernel: the constant kernel, represent $s_f^2$ in Eq. (20).

### Variables:  
* `x, y`: training data;
* `xL`: the lower limit of interpolation region;
* `xU`: the upper limit of interpolation region;
* `n`: the number of points between `xL` and `xU`;
* `test_x`: the test set;
* `test_y`: the output of the variable `test_x` that need to be predicted.

### Input file and input data: 
    feos = np.loadtxt('eos.dat',dtype=np.float) 
    x, y = feos[:,0], feos[:,1] 
    y = np.log10(y)  
* `"eos.dat"` is the input file. The first column in `"eos.dat"` is the baryon number density, the second column is the pressure. The first half of the data is the EoS of the hadron phase, and the second half is the EoS of quark phase.   
*  The order of magnitude between the input values `x` and `y`  shouldn't differ too much, otherwise, take the logarithm of the input values with large order of magnitude.

### Crossover window and steps:
    xL, xU, n = 0.3, 0.6, 100 
    test_x = np.linspace(xL, xU, n) 
* The interpolated baryon density ranges from 0.3 fm$^{-3}$ to 0.6 fm$^{-3}$ and 100 points are taken.

### Gaussian process regression:
    kernel = C(10, (1e-5, 1e4))*RBF(length_scale=1)+ WhiteKernel(noise_level=5e-3,noise_level_bounds=(1.8e-3, 3e-3)) 
    gp = GaussianProcessRegressor (kernel=kernel).fit(x[:, np.newaxis], y) 
    test_y, y_std = gp.predict(test_x[:, np.newaxis], return_std =True) 
*  `test_y` and `y_std` are the mean values and the standard deviations of the predicted points, respectively.

### 95% confidence interval:
    y_uncertainty = 1.96*y_std 

### Predicted points and error bar:
    test_yL, test_yU=10**(test_y-y_uncertainty), 10**(test_y+y_uncertainty)
    test_y=10**(test_y)

## References: 
* http://www.gaussianprocess.org/gpml/
* https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.RBF.html
* https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.WhiteKernel.html
* https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.ConstantKernel.html

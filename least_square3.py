from scipy.optimize import least_squares 
import numpy as np
import matplotlib.pyplot as plt

def generate_data(t, A, sigma,  noise=0, n_outliers=0, random_state=0): # The funtion for generating data
    y = A * np.exp(-sigma * t) 
    rnd = np.random.RandomState(random_state)
    error = noise * rnd.randn(t.size)
    outliers = rnd.randint(0, t.size, n_outliers)
    error[outliers] *= 35
    return y + error

# Model Parmeters
A = 2
sigma = 0.1
omega = 0.1 * 2 * np.pi
x_true = np.array([A, sigma])
noise = 0.1
t_min = 0
t_max = 30
#------------------
# Now, generate the data
t_train = np.linspace(t_min, t_max, 30)
y_train = generate_data(t_train, A, sigma,  noise=noise, n_outliers=4)
#------------------


# Define  a function to
# compute the residual 
# between the model and the data

def fun(x, t, y): # y = data, x[0] = A, x[1] = sigma
    return x[0] * np.exp(-x[1] * t)  - y

'''
Now least squares funtion take initial starting point 
guess for the free parameters in the residual function,
we have 2 : A, sigma, we define their starting point
as 1,1     defined below : 
'''

x0 = np.ones(2) 

'''Computing the least squares(2 ways) using scipy fun'''

res_lsq = least_squares(fun, x0, args=(t_train, y_train)) # 1st method 
res_lsq_bound = least_squares(fun, x0, args=(t_train, y_train),bounds=([-10, -1], [10, 1])) # 1st method + with bounds on parameter a[0],a[1]
res_robust = least_squares(fun, x0, loss='soft_l1', f_scale=0.1, args=(t_train, y_train)) # 2nd method 
#----------------------------------------
# new set of data for least square curve fitting

t_test = np.linspace(t_min, t_max, 300)# same as t_train but N = 300 points
y_test = generate_data(t_test, A, sigma)
y_lsq = generate_data(t_test, *res_lsq.x)
y_lsq_b = generate_data(t_test, *res_lsq_bound.x)
y_robust = generate_data(t_test, *res_robust.x)
 

#plt.plot(t_train, y_train,'o',c='r',  label='data')
plt.plot(t_test, y_test, label='true')
plt.plot(t_test, y_lsq, label='lsq')
plt.plot(t_test,y_lsq_b,label='lsq_b')
plt.plot(t_test, y_robust, label='robust lsq')
plt.xlabel('$t$')
plt.ylabel('$y$')
plt.legend();
plt.show() 

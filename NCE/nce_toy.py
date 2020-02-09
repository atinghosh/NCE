import jax.numpy as np
import numpy as onp
from jax.scipy.stats import norm
import jax
from jax import random


mu, sigma = 2, 4
noise_mu, noise_sigma = 0, 20
no_of_sample = 10000

inputs = onp.random.normal(mu, sigma, no_of_sample)
noise = onp.random.normal(noise_mu, noise_sigma, no_of_sample)

def logistic_fn(x):
    return 1/(1+np.exp(-x))

def log_unlormalized_pdf(x, mu_param, log_sigma_param, log_c_param):
    sigma = np.exp(log_sigma_param)
    return  log_c_param-(x-mu_param)**2/(2*sigma**2)
#     return log_c_param - (x-mu_param)**2/(2*sigma**2)

def obj_fun(mu_param, log_sigma_param, log_c_param):
    global inputs
    G_input = log_unlormalized_pdf(inputs, mu_param, log_sigma_param, log_c_param) - np.log(norm.pdf(inputs, noise_mu, noise_sigma))
    G_noise = log_unlormalized_pdf(noise, mu_param, log_sigma_param, log_c_param) - np.log(norm.pdf(noise, noise_mu, noise_sigma))
    h_input = logistic_fn(G_input)
    h_noise = logistic_fn(G_noise)
    
    loss = np.log(h_input) + np.log(1-h_noise)
    return -.5 * (1/no_of_sample)  * np.sum(loss)
    
loss_grad = jax.grad(obj_fun, argnums=(0,1,2))

mu_init, log_sigma_init, log_c_init = 20.0, 8.0, -3.0
params = mu_init, log_sigma_init, log_c_init
lr = .2
for ep in range(1000):
    loss = obj_fun(*params)
    grads = loss_grad(*params)

    params = [param - lr * grad for param, grad in zip(params, grads)]
    if ep % 50 ==0:
        print(f'Epoch is {ep}.......')
        print('LOSS')
        print(loss)
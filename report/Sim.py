import numpy as np
import matplotlib.pyplot as plt

iterations = 10000
n = 30
secret_beta = 2

var_u_list = [0, 1.5, 2, 2.5,3.0,4.5,6.0,8.0,10.0]
var_eps_list = [1.5, 2, 2.5]

var_u_list = [0, 1.5, 2, 2.5,3.0,4.5,6.0,8.0,10.0]
var_eps_list = [1.5, 2, 2.5]

def experiment(distribution_label, gen_err_u, gen_err_eps):
    print(f"{distribution_label} distributed epsilon")
    
    beta_dict = {}
    sigma_dict = {}

    for var_u in var_u_list:
        for var_eps in var_eps_list:
            bias_beta_ols = np.array([])
            sqerr_beta_ols = np.array([])
            bias_beta_mme = np.array([])
            sqerr_beta_mme = np.array([])

            bias_sigma_ols = np.array([])
            sqerr_sigma_ols = np.array([])
            bias_sigma_mme = np.array([])
            sqerr_sigma_mme = np.array([])

            for _ in range(iterations):
                x = np.linspace(-20,20,n)
                if var_u > 0:
                    x_obs = x + gen_err_u(var_u, n)
                else:
                    x_obs = x
                y = x*secret_beta + gen_err_x(var_eps, n)

                beta_ols = np.cov(x_obs, y, ddof=0)[0][1]/np.var(x_obs, ddof=0)
                beta_mme = np.cov(x_obs, y, ddof=1)[0][1]/(np.var(x_obs, ddof=1) - var_u)
                
                vx = np.var(x_obs, ddof=0) - var_u
                vu = var_u
                l = vx/(vx+vu);

                sigma_ols = np.sum((y - beta_ols * x_obs) ** 2) / (n-1)
                sigma_mme = np.sum((y - beta_ols * x_obs) ** 2) / (n-1) - (1-l)**2*beta_mme**2*vx - l**2*beta_mme**2*vu
                
                bias_beta_ols = np.append(bias_beta_ols, (beta_ols-secret_beta))
                sqerr_beta_ols = np.append(sqerr_beta_ols, (beta_ols-secret_beta)**2)
                bias_beta_mme = np.append(bias_beta_mme, (beta_mme-secret_beta))
                sqerr_beta_mme = np.append(sqerr_beta_mme, (beta_mme-secret_beta)**2)

                bias_sigma_ols = np.append(bias_sigma_ols, sigma_ols-var_eps)
                sqerr_sigma_ols = np.append(sqerr_sigma_ols, (sigma_ols-var_eps)**2)
                bias_sigma_mme = np.append(bias_sigma_mme, sigma_mme-var_eps)
                sqerr_sigma_mme = np.append(sqerr_sigma_mme, (sigma_mme-var_eps)**2)

            print("var_u=%.2f, var_eps=%.2f:  OLS: MBE = %f, MSE = %f; Sigma MBE = %f, MSE = %f"%(var_u, var_eps, bias_beta_ols.mean(), sqerr_beta_ols.mean(), bias_sigma_ols.mean(), sqerr_sigma_ols.mean()))

            print("var_u=%.2f, var_eps=%.2f:  MME: MBE = %f, MSE = %f; Sigma MBE = %f, MSE = %f"%(var_u, var_eps, bias_beta_mme.mean(), sqerr_beta_mme.mean(), bias_sigma_mme.mean(), sqerr_sigma_mme.mean()))
        
            beta_dict[(var_u, var_eps)] = {}
            beta_dict[(var_u, var_eps)]['bias_ols'] = bias_beta_ols.mean()
            beta_dict[(var_u, var_eps)]['sqerr_ols'] = sqerr_beta_ols.mean()
            beta_dict[(var_u, var_eps)]['bias_mme'] = bias_beta_mme.mean()
            beta_dict[(var_u, var_eps)]['sqerr_mme'] = sqerr_beta_mme.mean()
            
            sigma_dict[(var_u, var_eps)] = {}
            sigma_dict[(var_u, var_eps)]['bias_ols'] = bias_sigma_ols.mean()
            sigma_dict[(var_u, var_eps)]['sqerr_ols'] = sqerr_sigma_ols.mean()
            sigma_dict[(var_u, var_eps)]['bias_mme'] = bias_sigma_mme.mean()
            sigma_dict[(var_u, var_eps)]['sqerr_mme'] = sqerr_sigma_mme.mean()
            
        
    return beta_dict, sigma_dict

# experiment under normal-distributed errors

def gen_err_x(var, n):
    return np.random.normal(scale=np.sqrt(var), size=n)

def gen_err_eps(var, n):
    return np.random.normal(scale=np.sqrt(var), size=n)

beta_dict_normal, sigma_dict_normal = experiment("Normal", gen_err_x, gen_err_eps)

# experiment under Student's t-distributed errors

def gen_err_x(var, n):
    return np.random.standard_t(df=2*var/(var-1), size=n) # var=df/(df-2)

def gen_err_eps(var, n):
    return np.random.standard_t(df=2*var/(var-1), size=n)

beta_dict_t, sigma_dict_t = experiment("Student-T", gen_err_x, gen_err_eps)

# experiment under re-centered Chi^2-distributed errors

def gen_err_x(var, n):
    return np.random.chisquare(df=var/2.0, size=n) - var/2.0# var=2*dof

def gen_err_eps(var, n):
    return np.random.chisquare(df=var/2.0, size=n) - var/2.0

beta_dict_chi, sigma_dict_chi = experiment("ChiSq", gen_err_x, gen_err_eps)
var_u_list = [0, 1.5, 2, 2.5,3.0,4.5,6.0,8.0,10.0]
var_eps_list = [1.5, 2, 2.5]

def experiment(distribution_label, gen_err_u, gen_err_eps):
    print(f"{distribution_label} distributed epsilon")

    beta_dict = {}
    sigma_dict = {}

    for var_u in var_u_list:
        for var_eps in var_eps_list:
            bias_beta = np.array([])
            sqerr_beta = np.array([])
            bias_beta_adj = np.array([])
            sqerr_beta_adj = np.array([])

            bias_sigma = np.array([])
            sqerr_sigma = np.array([])
            bias_sigma_adj_v2 = np.array([])
            sqerr_sigma_adj_v2 = np.array([])

            for _ in range(iterations):
                x = np.linspace(-20,20,n)
                if var_u > 0:
                    x_obs = x + gen_err_u(var_u, n)
                else:
                    x_obs = x
                y = x*secret_beta + gen_err_x(var_eps, n)

                beta_est = np.cov(x_obs, y)[0][1]/np.var(x_obs, ddof=1)
                beta_est_adj = np.cov(x_obs, y)[0][1]/np.var(x_obs, ddof=1) * np.var(x_obs, ddof=1)/(np.var(x_obs, ddof=1) - var_u)

                vx = np.var(x_obs, ddof=1) - var_u
                vu = var_u
                l = vx/(vx+vu);

                sigma_est = np.sum((y - beta_est * x_obs) ** 2) / (n-1)
                sigma_est_adj_v2 = np.sum((y - beta_est * x_obs) ** 2) / (n-1) - (1-l)**2*beta_est_adj**2*vx - l**2*beta_est_adj**2*vu

                bias_beta = np.append(bias_beta, (beta_est-secret_beta))
                sqerr_beta = np.append(sqerr_beta, (beta_est-secret_beta)**2)
                bias_beta_adj = np.append(bias_beta_adj, (beta_est_adj-secret_beta))
                sqerr_beta_adj = np.append(sqerr_beta_adj, (beta_est_adj-secret_beta)**2)

                bias_sigma = np.append(bias_sigma, sigma_est-var_eps)
                sqerr_sigma = np.append(sqerr_sigma, (sigma_est-var_eps)**2)
                bias_sigma_adj_v2 = np.append(bias_sigma_adj_v2, sigma_est_adj_v2-var_eps)
                sqerr_sigma_adj_v2 = np.append(sqerr_sigma_adj_v2, (sigma_est_adj_v2-var_eps)**2)

            print("var_u=%.2f, var_eps=%.2f:  OLS: MBE = %f, MSE = %f; Sigma MBE = %f, MSE = %f"%(var_u, var_eps, bias_beta.mean(), sqerr_beta.mean(), bias_sigma.mean(), sqerr_sigma.mean()))

            print("var_u=%.2f, var_eps=%.2f:  MME: MBE = %f, MSE = %f; Sigma MBE = %f, MSE = %f"%(var_u, var_eps, bias_beta_adj.mean(), sqerr_beta_adj.mean(), bias_sigma_adj_v2.mean(), sqerr_sigma_adj_v2.mean()))

            beta_dict[(var_u, var_eps)] = {}
            beta_dict[(var_u, var_eps)]['bias_ols'] = bias_beta.mean()
            beta_dict[(var_u, var_eps)]['sqerr_ols'] = sqerr_beta.mean()
            beta_dict[(var_u, var_eps)]['bias_ols_corr'] = bias_beta_adj.mean()
            beta_dict[(var_u, var_eps)]['sqerr_ols_corr'] = sqerr_beta_adj.mean()

            sigma_dict[(var_u, var_eps)] = {}
            sigma_dict[(var_u, var_eps)]['bias_ols'] = bias_sigma.mean()
            sigma_dict[(var_u, var_eps)]['sqerr_ols'] = sqerr_sigma.mean()
            sigma_dict[(var_u, var_eps)]['bias_ols_corr_v2'] = bias_sigma_adj_v2.mean()
            sigma_dict[(var_u, var_eps)]['sqerr_ols_corr_v2'] = sqerr_sigma_adj_v2.mean()


    return beta_dict, sigma_dict

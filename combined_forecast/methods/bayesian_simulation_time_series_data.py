import numpy as np
from scipy import stats
import numpy.linalg as la
from scipy.special import gammaln, gammaincc
from scipy.optimize import minimize  # For gradient-based optimization
import matplotlib.pyplot as plt  # For plotting

# -----------------------------------------------------------
# Helper functions

def sample_inv_gauss(mu, lam):
    v = np.random.normal()
    y = v**2
    x = mu + (mu**2 * y)/(2*lam) - (mu/(2*lam))*np.sqrt(4*mu*lam*y + mu**2 * y**2)
    u = np.random.rand()
    return x if u <= mu/(mu + x) else (mu**2)/x

def sample_sigma2_li_lin(lambda1, lambda2, p, shape_param, scale_param, max_iter=10000):
    log_gamma_half = gammaln(0.5)
    for _ in range(max_iter):
        Z = stats.invgamma.rvs(a=shape_param, scale=scale_param)
        u = np.random.rand()
        upper_gamma_arg = (lambda1**2)/(8 * Z * lambda2)
        Gamma_U = gammaincc(0.5, upper_gamma_arg) * np.exp(log_gamma_half)
        log_accept_ratio = p * log_gamma_half - p * np.log(Gamma_U)
        if np.log(u) <= log_accept_ratio:
            return Z
    raise RuntimeError("Sampling σ² via AR failed after max_iter iterations.")

# -----------------------------------------------------------
# MCEM for λ₁, λ₂ tuning

def Q_function(lambda1, lambda2, beta_samples, sigma2_samples, u_samples):
    N, p = beta_samples.shape
    total = 0.0
    for i in range(N):
        sigma2 = sigma2_samples[i]
        beta_i = beta_samples[i]
        u_i = u_samples[i]
        tau = 1 + 1/u_i
        arg = lambda1**2/(8 * sigma2 * lambda2)
        U = gammaincc(0.5, arg)
        U = np.clip(U, 1e-300, 1.0)
        term1 = p * np.log(lambda1)
        term2 = -p * (np.log(U) + gammaln(0.5))
        term3 = -(lambda2/2) * np.sum((tau/(tau-1)) * (beta_i**2)/sigma2)
        term4 = -(lambda1**2/(8*lambda2)) * np.sum(tau/sigma2)
        total += term1 + term2 + term3 + term4
    return total/N

def maximize_Q_function_gradient(beta_samples, sigma2_samples, u_samples, init_params):
    def obj(params):
        l1, l2 = params
        if l1 <= 0 or l2 <= 0:
            return np.inf
        return -Q_function(l1, l2, beta_samples, sigma2_samples, u_samples)
    bounds = [(0.1, None), (0.1, None)]
    result = minimize(obj, init_params, method='L-BFGS-B', bounds=bounds)
    if not result.success:
        raise RuntimeError("Lambda optimization failed: " + result.message)
    return result.x[0], result.x[1]

def bayesian_elastic_net_inner(X, y, num_iter=10000, burn_in=5000, lambda1=1.0, lambda2=1.0):
    n, p = X.shape
    beta = np.zeros(p)
    sigma2 = 1.0
    u = np.ones(p)
    beta_samples, sigma2_samples, u_samples = [], [], []
    XtX, Xty = X.T @ X, X.T @ y
    for it in range(num_iter):
        D = np.diag(lambda2 * (1 + u))
        cov_beta = la.inv(XtX + D)
        cov_beta = (cov_beta + cov_beta.T) / 2
        mu_beta = cov_beta @ Xty
        beta = np.random.multivariate_normal(mu_beta, sigma2 * cov_beta)
        for j in range(p):
            if abs(beta[j]) < 1e-8:
                u[j] = 1e-2
            else:
                mu_ig = np.sqrt(lambda1)/(2*lambda2*abs(beta[j]))
                lam_ig = lambda1/(4*lambda2*sigma2)
                u[j] = max(sample_inv_gauss(mu_ig, lam_ig), 1e-2)
        resid = y - X @ beta
        rss = np.sum(resid**2)
        penalty_beta = lambda2 * np.sum((1 + u) * beta**2)
        penalty_latent = (lambda1**2/(4*lambda2)) * np.sum(1 + 1/u)
        shape_param = n/2 + p
        scale_param = 0.5 * (rss + penalty_beta + penalty_latent)
        sigma2 = sample_sigma2_li_lin(lambda1, lambda2, p, shape_param, scale_param)
        if it >= burn_in:
            beta_samples.append(beta.copy())
            sigma2_samples.append(sigma2)
            u_samples.append(u.copy())
    return np.array(beta_samples), np.array(sigma2_samples), np.array(u_samples)

def bayesian_elastic_net_EM(X, y, num_em_iter=10, inner_iter=10000, burn_in=5000,
                            init_lambda1=1.0, init_lambda2=1.0):
    lambda1, lambda2 = init_lambda1, init_lambda2
    print(f"Initial λ₁={lambda1:.3f}, λ₂={lambda2:.3f}")
    for em in range(num_em_iter):
        bs, s2s, us = bayesian_elastic_net_inner(
            X, y, num_iter=inner_iter, burn_in=burn_in,
            lambda1=lambda1, lambda2=lambda2
        )
        lambda1, lambda2 = maximize_Q_function_gradient(bs, s2s, us, [lambda1, lambda2])
        print(f"EM iter {em+1}: λ₁={lambda1:.3f}, λ₂={lambda2:.3f}")
    bs, s2s, _ = bayesian_elastic_net_inner(
        X, y, num_iter=inner_iter, burn_in=burn_in,
        lambda1=lambda1, lambda2=lambda2
    )
    return bs, s2s, lambda1, lambda2

def bayesian_elastic_net(X, y, num_iter=2000, burn_in=1000, lambda1=1.0, lambda2=1.0):
    n, p = X.shape
    beta = np.zeros(p)
    sigma2 = 1.0
    u = np.ones(p)
    beta_samples, sigma2_samples = [], []
    XtX, Xty = X.T @ X, X.T @ y
    for it in range(num_iter):
        D = np.diag(lambda2 * (1 + u))
        cov_beta = la.inv(XtX + D)
        cov_beta = (cov_beta + cov_beta.T) / 2
        mu_beta = cov_beta @ Xty
        beta = np.random.multivariate_normal(mu_beta, sigma2 * cov_beta)
        for j in range(p):
            if abs(beta[j]) < 1e-8:
                u[j] = 1e-2
            else:
                mu_ig = np.sqrt(lambda1)/(2*lambda2*abs(beta[j]))
                lam_ig = lambda1/(4*lambda2*sigma2)
                u[j] = max(sample_inv_gauss(mu_ig, lam_ig), 1e-2)
        resid = y - X @ beta
        rss = np.sum(resid**2)
        penalty_beta = lambda2 * np.sum((1 + u) * beta**2)
        penalty_latent = (lambda1**2/(4*lambda2)) * np.sum(1 + 1/u)
        shape_param = n/2 + p
        scale_param = 0.5 * (rss + penalty_beta + penalty_latent)
        sigma2 = sample_sigma2_li_lin(lambda1, lambda2, p, shape_param, scale_param)
        if it >= burn_in:
            beta_samples.append(beta.copy())
            sigma2_samples.append(sigma2)
    return np.array(beta_samples), np.array(sigma2_samples)

# -----------------------------------------------------------
# Main: sliding window with daily percent‑outside printing

def main():
    np.random.seed(42)
    total_days = int(365 * 1.5)    # 547 days
    hours_per_day = 24
    n = total_days * hours_per_day
    t = np.arange(n)

    # deterministic covariates
    x1 = t
    x2 = np.sin(2 * np.pi * t / hours_per_day)
    x3 = np.cos(2 * np.pi * t / hours_per_day)
    x4 = np.sin(2 * np.pi * t / (hours_per_day * 7))
    x5 = np.cos(2 * np.pi * t / (hours_per_day * 7))
    X = np.column_stack([x1, x2, x3, x4, x5])

    beta_true = np.array([0.0001, 3.0, -3.0, 1.5, -1.5])
    sigma = 1.0
    y = X.dot(beta_true) + np.random.normal(scale=sigma, size=n)

    window_days = 165
    horizon_days = 16
    days_for_forecast = total_days - window_days
    num_blocks = days_for_forecast // horizon_days

    daily_perc = []

    # Loop over blocks, retune lambdas every horizon_days
    for b in range(num_blocks + 1):
        start_lambda = b * horizon_days
        end_lambda = start_lambda + window_days
        if end_lambda > total_days:
            break

        # tune lambdas on the current window
        i0, i1 = start_lambda * hours_per_day, end_lambda * hours_per_day
        X_l, y_l = X[i0:i1], y[i0:i1]
        _, _, l1, l2 = bayesian_elastic_net_EM(X_l, y_l,
                                              num_em_iter=10,
                                              inner_iter=10000,
                                              burn_in=5000)

        # forecast each day with fixed lambdas
        for d in range(horizon_days):
            train_start = start_lambda + d
            train_end   = train_start + window_days
            if train_end >= total_days:
                break

            te_day = train_end
            tr0, tr1 = train_start * hours_per_day, train_end * hours_per_day
            te0, te1 = te_day * hours_per_day, (te_day + 1) * hours_per_day

            X_tr, y_tr = X[tr0:tr1], y[tr0:tr1]
            X_te, y_te = X[te0:te1], y[te0:te1]

            # draw from posterior and compute predictive bands
            beta_samps, s2_samps = bayesian_elastic_net(
                X_tr, y_tr,
                num_iter=2000, burn_in=1000,
                lambda1=l1, lambda2=l2
            )

            n_post = len(beta_samps)
            preds = np.zeros((hours_per_day, n_post))
            for i in range(n_post):
                preds[:, i] = (
                    X_te.dot(beta_samps[i])
                    + np.random.normal(scale=np.sqrt(s2_samps[i]), size=hours_per_day)
                )

            low  = np.percentile(preds, 2.5, axis=1)
            high = np.percentile(preds, 97.5, axis=1)
            outside = np.sum((y_te < low) | (y_te > high))

            # compute percent outside and print for this day
            pct_out = outside / hours_per_day * 100
            daily_perc.append(pct_out)
            print(f"Day {te_day:3d}: {pct_out:.2f}% of hours outside the 95% interval")

    # aggregate into weeks and plot overall coverage
    hours_per_week = 7 * hours_per_day
    weekly_perc = []
    for i in range(0, len(daily_perc), 7):
        week_days = daily_perc[i:i+7]
        weekly_perc.append(np.mean(week_days))

    # --- final plot with y-axis from 0 to 20 and ticks every 5 ---
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(1, len(weekly_perc) + 1), weekly_perc, marker='o')
    plt.xlabel('Week Index')
    plt.ylabel('Avg. Daily % Outside 95% Interval')
    plt.title('Weekly Predictive Interval Coverage')
    plt.ylim(0, 20)
    plt.yticks(np.arange(0, 21, 5))
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()

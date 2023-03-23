import logging

import numpy as np
import pandas as pd
import scipy
from scipy.special import expit, logit

logger = logging.getLogger("causaldata")

"""
上述代码中定义了五个函数，分别对应了五种不同的生成模拟数据的方法：

simulate_nuisance_and_easy_treatment()：这种方法生成的数据，包含了难以处理的干扰项和容易估计的处理效应。它是根据 Nie X. 和 Wager S. (2018) 的 "Quasi-Oracle Estimation of Heterogeneous Treatment Effects" 中的 Setup A 生成的。

simulate_randomized_trial()：这种方法生成的数据，模拟了一个随机试验。它是根据 Nie X. 和 Wager S. (2018) 的 "Quasi-Oracle Estimation of Heterogeneous Treatment Effects" 中的 Setup B 生成的。

simulate_easy_propensity_difficult_baseline()：这种方法生成的数据，模拟了一个易于估计的倾向得分和一个难以估计的基线。它是根据 Nie X. 和 Wager S. (2018) 的 "Quasi-Oracle Estimation of Heterogeneous Treatment Effects" 中的 Setup C 生成的。

simulate_unrelated_treatment_control()：这种方法生成的数据，模拟了一个不相关的处理和对照组。它是根据 Nie X. 和 Wager S. (2018) 的 "Quasi-Oracle Estimation of Heterogeneous Treatment Effects" 中的 Setup D 生成的。

simulate_hidden_confounder()：这种方法生成的数据，模拟了一个隐藏的混淆变量对处理造成偏倚的情况。它是根据 Louizos et al. (2018) 的 "Causal Effect Inference with Deep Latent-Variable Models" 生成的。

这些函数的输出都是一个包含六个元素的元组，分别为 y（因变量）、X（自变量）、w（处理）、tau（处理效应）、b（期望结果）和 e（处理得分）。

通过选择不同的 mode 参数值，我们可以使用这些函数生成不同的模拟数据，用于评估不同的因果推断方法的性能。
"""
def to_dataframe(func):
    def wrapper(*args, **kwargs):
        output_dataframe = kwargs.pop("to_dataframe", False)
        tmp = func(*args, **kwargs)
        if output_dataframe:
            y, X, w, tau, b, e = tmp
            df = pd.DataFrame(X)
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            df.columns = feature_names
            df['outcome'] = y
            df['treatment'] = w
            df['treatment_effect'] = tau
            return df
        else:
            return tmp
    return wrapper
@to_dataframe
def synthetic_data(mode=1, n=1000, p=5, sigma=1.0, adj=0.0):
    """ Synthetic data in Nie X. and Wager S. (2018) 'Quasi-Oracle Estimation of Heterogeneous Treatment Effects'
    Args:
        mode (int, optional): mode of the simulation: \
            1 for difficult nuisance components and an easy treatment effect. \
            2 for a randomized trial. \
            3 for an easy propensity and a difficult baseline. \
            4 for unrelated treatment and control groups. \
            5 for a hidden confounder biasing treatment.
        n (int, optional): number of observations
        p (int optional): number of covariates (>=5)
        sigma (float): standard deviation of the error term
        adj (float): adjustment term for the distribution of propensity, e. Higher values shift the distribution to 0.
                     It does not apply to mode == 2 or 3.
    Returns:
        (tuple): Synthetically generated samples with the following outputs:
            - y ((n,)-array): outcome variable.
            - X ((n,p)-ndarray): independent variables.
            - w ((n,)-array): treatment flag with value 0 or 1.
            - tau ((n,)-array): individual treatment effect.
            - b ((n,)-array): expected outcome.
            - e ((n,)-array): propensity of receiving treatment.
    """

    catalog = {
        1: simulate_nuisance_and_easy_treatment,
        2: simulate_randomized_trial,
        3: simulate_easy_propensity_difficult_baseline,
        4: simulate_unrelated_treatment_control,
        5: simulate_hidden_confounder,
    }

    assert mode in catalog, "Invalid mode {}. Should be one of {}".format(
        mode, set(catalog)
    )
    return catalog[mode](n, p, sigma, adj)


def synthetic_data_advanced(mode=1, n=1000, p=5, sigma=1.0, adj=0.0):
    """ 更加复杂的合成数据，e.g. multitreatment, continuous treatment, etc."""
    catalog = {
        1: multitreatment_base,
        2: simulate_randomized_trial,
        3: simulate_foo,
        4: simulate_unrelated_treatment_control,
        5: simulate_hidden_confounder,
    }

    assert mode in catalog, "Invalid mode {}. Should be one of {}".format(
        mode, set(catalog)
    )
    return catalog[mode](n, p, sigma, adj)

@to_dataframe
def synthetic_iv_data(mode=1, n=1000, p=5, sigma=1.0, adj=0.0):
    """ Synthetic IV data
    Args:
        mode (int, optional): mode of the simulation: \
            1 ...
            2 ...
            3 工具变量连续 treatment
        n (int, optional): number of observations
        p (int optional): number of covariates (>=5)
        sigma (float): standard deviation of the error term
        adj (float): adjustment term for the distribution of propensity, e. Higher values shift the distribution to 0.
                     It does not apply to mode == 2 or 3.
    Returns:
        (tuple): Synthetically generated samples with the following outputs:
            - y ((n,)-array): outcome variable.
            - X ((n,p)-ndarray): independent variables.
            - w ((n,)-array): treatment flag with value 0 or 1. ---> ....
            - tau ((n,)-array): individual treatment effect.
            - b ((n,)-array): expected outcome.
            - e ((n,)-array): propensity of receiving treatment.
    """

    catalog = {
        1: simulate_nuisance_and_easy_treatment,
        2: simulate_randomized_trial,
        3: simulate_continuous_treatment,
        4: foo,
        5: simulate_hidden_confounder,
    }

    assert mode in catalog, "Invalid mode {}. Should be one of {}".format(
        mode, set(catalog)
    )
    return catalog[mode](n, p, sigma, adj)


## TODO 半合成数据
@to_dataframe
def semi_synthetic_data(mode=1, n=1000, p=5, sigma=1.0, adj=0.0, dataframe=False):
    """ 半合成数据"""
    catalog = {
        1: simulate_nuisance_and_easy_treatment,
        2: simulate_randomized_trial,
        3: simulate_easy_propensity_difficult_baseline,
        4: simulate_unrelated_treatment_control,
        5: simulate_hidden_confounder,
    }

    assert mode in catalog, "Invalid mode {}. Should be one of {}".format(
        mode, set(catalog)
    )
    return catalog[mode](n, p, sigma, adj)

@to_dataframe
def simulate_nuisance_and_easy_treatment(n=1000, p=5, sigma=1.0, adj=0.0):
    """Synthetic data with a difficult nuisance components and an easy treatment effect
        From Setup A in Nie X. and Wager S. (2018) 'Quasi-Oracle Estimation of Heterogeneous Treatment Effects'
    Args:
        n (int, optional): number of observations
        p (int optional): number of covariates (>=5)
        sigma (float): standard deviation of the error term
        adj (float): adjustment term for the distribution of propensity, e. Higher values shift the distribution to 0.
    Returns:
        (tuple): Synthetically generated samples with the following outputs:
            - y ((n,)-array): outcome variable.
            - X ((n,p)-ndarray): independent variables.
            - w ((n,)-array): treatment flag with value 0 or 1.
            - tau ((n,)-array): individual treatment effect.
            - b ((n,)-array): expected outcome.
            - e ((n,)-array): propensity of receiving treatment.
    """

    X = np.random.uniform(size=n * p).reshape((n, -1))
    b = (
            np.sin(np.pi * X[:, 0] * X[:, 1])
            + 2 * (X[:, 2] - 0.5) ** 2
            + X[:, 3]
            + 0.5 * X[:, 4]
    )
    eta = 0.1
    e = np.maximum(
        np.repeat(eta, n),
        np.minimum(np.sin(np.pi * X[:, 0] * X[:, 1]), np.repeat(1 - eta, n)),
    )
    e = expit(logit(e) - adj)
    tau = (X[:, 0] + X[:, 1]) / 2

    w = np.random.binomial(1, e, size=n)
    y = b + (w - 0.5) * tau + sigma * np.random.normal(size=n)

    return y, X, w, tau, b, e

@to_dataframe
def simulate_randomized_trial(n=1000, p=5, sigma=1.0, adj=0.0):
    """Synthetic data of a randomized trial
        From Setup B in Nie X. and Wager S. (2018) 'Quasi-Oracle Estimation of Heterogeneous Treatment Effects'
    Args:
        n (int, optional): number of observations
        p (int optional): number of covariates (>=5)
        sigma (float): standard deviation of the error term
        adj (float): no effect. added for consistency
    Returns:
        (tuple): Synthetically generated samples with the following outputs:
            - y ((n,)-array): outcome variable.
            - X ((n,p)-ndarray): independent variables.
            - w ((n,)-array): treatment flag with value 0 or 1.
            - tau ((n,)-array): individual treatment effect.
            - b ((n,)-array): expected outcome.
            - e ((n,)-array): propensity of receiving treatment.
    """

    X = np.random.normal(size=n * p).reshape((n, -1))
    b = np.maximum(np.repeat(0.0, n), X[:, 0] + X[:, 1], X[:, 2]) + np.maximum(
        np.repeat(0.0, n), X[:, 3] + X[:, 4]
    )
    e = np.repeat(0.5, n)
    tau = X[:, 0] + np.log1p(np.exp(X[:, 1]))

    w = np.random.binomial(1, e, size=n)
    y = b + (w - 0.5) * tau + sigma * np.random.normal(size=n)

    return y, X, w, tau, b, e


def simulate_easy_propensity_difficult_baseline(n=1000, p=5, sigma=1.0, adj=0.0):
    """Synthetic data with easy propensity and a difficult baseline
        From Setup C in Nie X. and Wager S. (2018) 'Quasi-Oracle Estimation of Heterogeneous Treatment Effects'
    Args:
        n (int, optional): number of observations
        p (int optional): number of covariates (>=3)
        sigma (float): standard deviation of the error term
        adj (float): no effect. added for consistency
    Returns:
        (tuple): Synthetically generated samples with the following outputs:
            - y ((n,)-array): outcome variable.
            - X ((n,p)-ndarray): independent variables.
            - w ((n,)-array): treatment flag with value 0 or 1.
            - tau ((n,)-array): individual treatment effect.
            - b ((n,)-array): expected outcome.
            - e ((n,)-array): propensity of receiving treatment.
    """

    X = np.random.normal(size=n * p).reshape((n, -1))
    b = 2 * np.log1p(np.exp(X[:, 0] + X[:, 1] + X[:, 2]))
    e = 1 / (1 + np.exp(X[:, 1] + X[:, 2]))
    tau = np.repeat(1.0, n)

    w = np.random.binomial(1, e, size=n)
    y = b + (w - 0.5) * tau + sigma * np.random.normal(size=n)

    return y, X, w, tau, b, e


def simulate_unrelated_treatment_control(n=1000, p=5, sigma=1.0, adj=0.0):
    """Synthetic data with unrelated treatment and control groups.
        From Setup D in Nie X. and Wager S. (2018) 'Quasi-Oracle Estimation of Heterogeneous Treatment Effects'
    Args:
        n (int, optional): number of observations
        p (int optional): number of covariates (>=3)
        sigma (float): standard deviation of the error term
        adj (float): adjustment term for the distribution of propensity, e. Higher values shift the distribution to 0.
    Returns:
        (tuple): Synthetically generated samples with the following outputs:
            - y ((n,)-array): outcome variable.
            - X ((n,p)-ndarray): independent variables.
            - w ((n,)-array): treatment flag with value 0 or 1.
            - tau ((n,)-array): individual treatment effect.
            - b ((n,)-array): expected outcome.
            - e ((n,)-array): propensity of receiving treatment.
    """

    X = np.random.normal(size=n * p).reshape((n, -1))
    b = (
                np.maximum(np.repeat(0.0, n), X[:, 0] + X[:, 1] + X[:, 2])
                + np.maximum(np.repeat(0.0, n), X[:, 3] + X[:, 4])
        ) / 2
    e = 1 / (1 + np.exp(-X[:, 0]) + np.exp(-X[:, 1]))
    e = expit(logit(e) - adj)
    tau = np.maximum(np.repeat(0.0, n), X[:, 0] + X[:, 1] + X[:, 2]) - np.maximum(
        np.repeat(0.0, n), X[:, 3] + X[:, 4]
    )

    w = np.random.binomial(1, e, size=n)
    y = b + (w - 0.5) * tau + sigma * np.random.normal(size=n)

    return y, X, w, tau, b, e


def simulate_hidden_confounder(n=10000, p=5, sigma=1.0, adj=0.0):
    """Synthetic dataset with a hidden confounder biasing treatment.
        From Louizos et al. (2018) "Causal Effect Inference with Deep Latent-Variable Models"
    Args:
        n (int, optional): number of observations
        p (int optional): number of covariates (>=3)
        sigma (float): standard deviation of the error term
        adj (float): no effect. added for consistency
    Returns:
        (tuple): Synthetically generated samples with the following outputs:
            - y ((n,)-array): outcome variable.
            - X ((n,p)-ndarray): independent variables.
            - w ((n,)-array): treatment flag with value 0 or 1.
            - tau ((n,)-array): individual treatment effect.
            - b ((n,)-array): expected outcome.
            - e ((n,)-array): propensity of receiving treatment.
    """
    z = np.random.binomial(1, 0.5, size=n).astype(np.double)
    X = np.random.normal(z, 5 * z + 3 * (1 - z), size=(p, n)).T
    e = 0.75 * z + 0.25 * (1 - z)
    w = np.random.binomial(1, e)
    b = expit(3 * (z + 2 * (2 * w - 2)))
    y = np.random.binomial(1, b)

    # Compute true ite tau for evaluation (via Monte Carlo approximation).
    t0_t1 = np.array([[0.0], [1.0]])
    y_t0, y_t1 = expit(3 * (z + 2 * (2 * t0_t1 - 2)))
    tau = y_t1 - y_t0
    return y, X, w, tau, b, e




def simulate_continuous_treatment(n=1000, p=5, binary_treatment=False):
    """Synthetic iv data with continuous treatment.
    References: https://github.com/1587causalai/EconML/blob/75b40b6b07ee8aa49e0be75057b54e2458158284/notebooks/OrthoIV%20and%20DRIV%20Examples.ipynb

    """
    X = np.random.normal(0, 1, size=(n, p))
    Z = np.random.binomial(1, 0.5, size=(n,))
    nu = np.random.uniform(0, 5, size=(n,))
    coef_Z = 0.8
    C = np.random.binomial(
        1, coef_Z * scipy.special.expit(0.4 * X[:, 0] + nu)
    )  # Compliers when recomended
    C0 = np.random.binomial(
        1, 0.006 * np.ones(X.shape[0])
    )  # Non-compliers when not recommended
    tmp_T = C * Z + C0 * (1 - Z)
    if not binary_treatment:
        cost = lambda X: 10 * X[:, 1] ** 2
        w = cost(X) * tmp_T
    else:
        w = tmp_T

    true_fn = lambda X: X[:, 0] + 0.5 * X[:, 1] + 0.5 * X[:, 2]
    tau = true_fn(X)

    y = (
            true_fn(X) * w  # 这里意味着 outcome 关于 treatment 是线性的
            + 2 * nu
            + 5 * (X[:, 3] > 0)
            + 0.1 * np.random.uniform(0, 1, size=(n,))
    )
    return y, X, w, tau, Z



def multitreatment_base(n_samples=1000, n_features=10, n_treatments=3, sigma=1.0, adj=0.0):
    """multitreatment_base

    References http://localhost:8888/notebooks/notebooks/Generalized%20Random%20Forests.ipynb
    """
    # true_te = lambda X: np.hstack([X[:, [0]]**2 + 1, np.ones((X.shape[0], n_treatments - 1))])
    # true_te = lambda X: np.hstack([X[:, [0]]>0, np.ones((X.shape[0], n_treatments - 1))])
    true_te = lambda X: np.hstack([(X[:, [0]] > 0) * X[:, [0]],
                                   np.ones((X.shape[0], n_treatments - 1)) * np.arange(1, n_treatments).reshape(1, -1)])
    X = np.random.normal(0, 1, size=(n_samples, n_features))
    T = np.random.normal(0, 1, size=(n_samples, n_treatments))
    for t in range(n_treatments):
        T[:, t] = np.random.binomial(1, scipy.special.expit(X[:, 0]))
    y = np.sum(true_te(X) * T, axis=1, keepdims=True) + np.random.normal(0, .5, size=(n_samples, 1))
    return y, X, T, true_te(X)
def foo():
    """
    Simulate data for the example in the docstring of the function
    Returns

    References https://github.com/1587causalai/EconML/blob/75b40b6b07ee8aa49e0be75057b54e2458158284/notebooks/Deep%20IV%20Examples.ipynb
    -------

    """
    n = 5000
    # Initialize exogenous variables; normal errors, uniformly distributed covariates and instruments
    e = np.random.normal(size=(n,))
    x = np.random.uniform(low=0.0, high=10.0, size=(n,))
    z = np.random.uniform(low=0.0, high=10.0, size=(n,))

    # Initialize treatment variable
    t = np.sqrt((x + 2) * z) + e

    # Outcome equation
    y = t*t / 10 - x*t / 10 + e

def simulate_foo():
    """
    Simulate data for the example in the docstring of the function
    Returns

    References
    https://chatgithub.com/py-why/EconML/blob/main/notebooks/AutomatedML/Automated%20Machine%20Learning%20For%20EconML.ipynb
    -------

    """

    import math

    # Treatment effect function
    def te(x):
        return np.sin(2 * math.pi * x[0]) / 2 + 0.5

    def g(x):
        return np.power(np.sin(x), 2)

    def m(x, nu=0., gamma=1.):
        return 0.5 / math.pi * (np.sinh(gamma)) / (np.cosh(gamma) - np.cos(x - nu))

    # vectorized g and m for applying to dataset
    vg = np.vectorize(g)
    vm = np.vectorize(m)

    # DGP constants
    np.random.seed(123)
    n = 10000
    n_w = 30
    support_size = 5
    n_x = 1
    # Outcome support
    support_Y = np.random.choice(np.arange(n_w), size=support_size, replace=False)
    coefs_Y = np.random.uniform(0, 1, size=support_size)
    epsilon_sample = lambda n: np.random.uniform(-1, 1, size=n)
    # Treatment support
    support_T = support_Y
    coefs_T = np.random.uniform(0, 1, size=support_size)
    eta_sample = lambda n: np.random.uniform(-1, 1, size=n)

    # Generate controls, covariates, treatments and outcomes
    W = np.random.normal(0, 1, size=(n, n_w))
    X = np.random.uniform(0, 1, size=(n, n_x))
    # Heterogeneous treatment effects
    TE = np.array([te(x_i) for x_i in X])

    T = vg(np.dot(W[:, support_T], coefs_T)) + eta_sample(n)
    Y = TE * T + vm(np.dot(W[:, support_Y], coefs_Y)) + epsilon_sample(n)
    return Y, T, X, W, TE


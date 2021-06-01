# Imports
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import sklearn

# import tigramite
import sys

from neurips2020.lpcmci import LPCMCI

sys.path.insert(1, "/Users/shawnxys/Development/tigramite")

from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI

# https://github.com/xiangyu-sun-789/tigramite#overview
"""
Conditional independence test 	        Assumptions
ParCorr 	                            univariate, continuous, linear Gaussian dependencies
GPDC / GPDCtorch 	                    univariate, continuous, additive dependencies
CMIknn 	                                multivariate, continuous, general dependencies
CMIsymb 	                            univariate, discrete/categorical dependencies
"""
from tigramite.independence_tests import ParCorr, GPDC, CMIknn, CMIsymb


def generate_data():
    seed = 7
    auto_coeff = 0.95
    coeff = 0.4
    T = 500

    def lin(x): return x

    links = {0: [((0, -1), auto_coeff, lin),
                 ((1, -1), coeff, lin)
                 ],
             1: [((1, -1), auto_coeff, lin),
                 ],
             2: [((2, -1), auto_coeff, lin),
                 ((3, 0), -coeff, lin),
                 ],
             3: [((3, -1), auto_coeff, lin),
                 ((1, -2), coeff, lin),
                 ],
             4: [((4, -1), auto_coeff, lin),
                 ((3, 0), coeff, lin),
                 ],
             5: [((5, -1), 0.5 * auto_coeff, lin),
                 ((6, 0), coeff, lin),
                 ],
             6: [((6, -1), 0.5 * auto_coeff, lin),
                 ((5, -1), -coeff, lin),
                 ],
             7: [((7, -1), auto_coeff, lin),
                 ((8, 0), -coeff, lin),
                 ],
             8: [],
             }

    # Specify dynamical noise term distributions, here unit variance Gaussians
    random_state = np.random.RandomState(seed)
    noises = noises = [random_state.randn for j in links.keys()]

    data, nonstationarity_indicator = pp.structural_causal_process(
        links=links, T=T, noises=noises, seed=seed)
    T, N = data.shape

    # Initialize dataframe object, specify variable names
    var_names = [r'$X^{%d}$' % j for j in range(N)]
    dataframe = pp.DataFrame(data, var_names=var_names)

    true_graph = pp.links_to_graph(links=links)

    return dataframe, true_graph, var_names


if __name__ == "__main__":
    dataframe, _, var_names = generate_data()

    tau_max = 1  # the number of lags

    pc_alpha = 0.01

    # TODO: use nonlinear conditional independence test
    cond_ind_test = ParCorr(significance='analytic')

    """
    ##########################
    ##### PCMCI+ #####
    ##########################
    """

    pcmci = PCMCI(
        dataframe=dataframe,
        cond_ind_test=cond_ind_test,
        verbosity=0)

    pcmci.verbosity = 0  # shows the results without showing steps

    results_pcmciplus = pcmci.run_pcmciplus(tau_min=0, tau_max=tau_max, pc_alpha=pc_alpha)

    graph_pcmciplus = results_pcmciplus['graph']
    val_matrix_pcmciplus = results_pcmciplus['val_matrix']

    """
    ##########################
    ##### LPCMCI #####
    ##########################
    """

    lpcmci = LPCMCI(
        dataframe=dataframe,
        cond_ind_test=cond_ind_test)

    _ = lpcmci.run_lpcmci(
        tau_max=tau_max,
        pc_alpha=pc_alpha,
        max_p_non_ancestral=3,
        n_preliminary_iterations=4,  # k, not the number of lags
        prelim_only=False,
        verbosity=0)

    graph_lpcmci = lpcmci.graph
    val_matrix_lpcmci = lpcmci.val_min_matrix

    """
    ##########################
    ##### Draw Graph #####
    ##########################
    """

    """
    In the time series graph, each entry in graph can be directly visualized. 
    Directed lagged or contemporaneous links are drawn as arrows and unoriented or 
    conflicting contemporaneous links as corresponding straight lines. In each case, 
    the link color refers to the MCI value in val_matrix. Also here, if val_matrix is 
    not already symmetric for contemporaneous values, the maximum absolute value is shown.
    
    `val_matrix` can be used to indicate the strength of links. 
    Set it to `None` if we want edges in the same colour.
    """

    # Plot time series graph for PCMCI+
    tp.plot_time_series_graph(
        figsize=(8, 8),
        node_size=0.05,
        val_matrix=val_matrix_pcmciplus,
        link_matrix=graph_pcmciplus,
        var_names=var_names,
        link_colorbar_label='MCI for PCMCI+',
    )
    plt.show()

    # Plot time series graph for LPCMCI
    tp.plot_time_series_graph(
        figsize=(8, 8),
        node_size=0.05,
        val_matrix=val_matrix_lpcmci,
        link_matrix=graph_lpcmci,
        var_names=var_names,
        link_colorbar_label='MCI for LPCMCI',
    )
    plt.show()

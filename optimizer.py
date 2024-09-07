import GPyOpt
from GPyOpt.methods import BayesianOptimization


def opt_GPyOpt(f, bounds, iter_tot):

    bounds_GPyOpt = [
        {
            "name": "var_" + str(i + 1),
            "type": "continuous",
            "domain": (-1, 1),
        }
        for i in range(len(bounds))
    ]

    myBopt = GPyOpt.methods.BayesianOptimization(f, domain=bounds_GPyOpt)
    myBopt.run_optimization(max_iter=iter_tot)

    return myBopt.x_opt, myBopt.fx_opt, myBopt

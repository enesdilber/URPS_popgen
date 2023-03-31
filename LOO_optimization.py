from scipy import optimize
import numpy as np


def LOO_optimization(
    momi,
    params,
    jsfs,
    transformed=False,
    out_key=None,
    out_value=None,
    method="trust-constr",
    **kwargs
):
    """Returns scipy optimizer results

    Args:
        momi: Momi object
        params: Params object
        jsfs: joint-sfs
        transformed: whether to use transformed variables
        out_key: Take var out
        out_value: value for out_key
        method: scipy.optimize method
        **kwargs: optimizer options
    """
    # Leave-one-out Optimization

    if out_key is not None:
        orig_train_it = params[out_key].train_it
        params.set_train(out_key, False)
        params.set(out_key, float(out_value))

    theta_train_dict = params.theta_train_dict(transformed)
    train_keys = tuple(theta_train_dict)
    train_vals = list(theta_train_dict.values())

    def obj_for_scipy(train_vals, train_keys=train_keys):
        theta_train_dict = {key: float(val) for key, val in zip(train_keys, train_vals)}

        val, grad = momi.negative_loglik_with_gradient(
            params, jsfs, theta_train_dict, transformed=transformed
        )

        return val, np.array([grad[i] for i in train_keys])

    optimizer_options = dict(**kwargs)

    if transformed:
        LinearConstraints = ()
    else:
        LinearConstraints = params._linear_constraints_for_scipy()

    # Running the scipy optimizer
    res = optimize.minimize(
        fun=obj_for_scipy,
        x0=train_vals,
        jac=True,
        method=method,
        options=optimizer_options,
        constraints=LinearConstraints
    )

    if out_key is not None:
        params.set_train(out_key, orig_train_it)
    res.x = dict(zip(train_keys, res.x))
    return res

import sys
sys.path.append("/nfs/turbo/lsa-jonth/eneswork/pyslurm")
import os

from momi3.MOMI import Momi
from momi3.Params import Params

import pickle
import demes

import numpy as np

from scipy import optimize
from time import sleep


def get_srun(proc):
    from pyslurm import Slurm
    slurm = Slurm(
        user='enes',
        path='/scratch/stats_dept_root/stats_dept/enes/',
        account='jonth0'
    )

    time = "0-5:30:00"
    # srun = slurm.batch(
    #     f'#time={time}'
    # )

    srun_GPU = slurm.batch(
        f'#time={time}',
        "#mem-per-cpu=None",
        "#nodes=1",
        "#cpus-per-task=1",
        "#mem=16G",
        "#gpus-per-node=1",
        "#partition=gpu",
        '#job-name="GPU_bs"',
        "module load cudnn",
        "module load cuda"
    )

    mem = 1500
    time = "0-10:00:00"
    srun_CPU = slurm.batch(
        f'#mem-per-cpu={mem}',
        '#cpus-per-task=10',
        f'#time={time}',
        '#job-name="CPU_bs"'
    )

    if proc == 'CPU':
        srun = srun_CPU
    else:
        srun = srun_GPU

    return srun


def get_demo():
	return demes.load('arc5.yaml')


def get_params(momi):
    params = Params(momi)
    params.set_train_all_etas(True)
    params.set_train('eta_0', False)
    params.set_train_all_pis(True)
    return params


if __name__ == "__main__":
    # For sending GL_jobs: python bootstrap.py out=/tmp/ mode=send_jobs njobs=50 proc=GPU or (CPU)
    # For sending bootstrap iter: python out=/tmp/ mode=run
    args = sys.argv[1:]

    arg_d = {}
    for arg in args:
        k, v = arg.split('=')
        arg_d[k] = v

    out_path = arg_d['out']
    model_name = 'arc5'
    assert os.path.exists(out_path)

    if arg_d['mode'] == 'send_jobs':
        srun = get_srun(arg_d['proc'])
        for i in range(int(arg_d['njobs'])):
            test = f'python bootstrap.py out={out_path} seed={i} mode=run'
            jobid = srun.run(test)
            print(f"{jobid}: Sent -- {test}")
            sleep(0.05)

    else:
        seed = int(arg_d['seed'])

        demo = get_demo()
        sampled_demes = demo.metadata['sampled_demes']
        sample_sizes = demo.metadata['sample_sizes']
        momi = Momi(demo, sampled_demes, sample_sizes, jitted=True)
        params = get_params(momi)
        bounds = momi.bound_sampler(params, 1000, seed=108)
        momi = momi.bound(bounds)

        jsfs = np.load('jsfs_UNIF_Papuan_Sardinian_YRI_Vindija_Denisovan_50_108.npy')
        if seed != 0:
            jsfs = momi._bootstrap_sample(jsfs, seed=seed)

        transformed = True
        theta_train_dict = params.theta_train_dict(transformed)
        train_keys = tuple(theta_train_dict)
        train_vals = list(theta_train_dict.values())

        def obj_for_scipy(train_vals, train_keys=train_keys):
            theta_train_dict = {key: float(val) for key, val in zip(train_keys, train_vals)}

            val, grad = momi.negative_loglik_with_gradient(
                params, jsfs, theta_train_dict, transformed=transformed
            )

            return val, np.array([grad[i] for i in train_keys])

        res = {}

        res['tnc'] = optimize.minimize(
            fun=obj_for_scipy,
            x0=train_vals,
            jac=True,
            method='TNC'
        )

        res['slsqp'] = optimize.minimize(
            fun=obj_for_scipy,
            x0=train_vals,
            jac=True,
            method='SLSQP'
        )

        for lr in [0.5, 0.05, 0.005]:
            res[lr] = optimize.minimize(
                fun=obj_for_scipy,
                x0=train_vals,
                jac=True,
                method='trust-constr',
                options=dict(initial_tr_radius=lr)
            )

        min_loss = np.inf
        for i in res:
            if res[i].fun < min_loss:
                min_loss = res[i].fun
                opt_x = res[i].x
                opt_res = res[i]

        opt_params = dict(zip(train_keys, opt_x))
        params.set_optimization_results(opt_params)
        opt_out = params.theta_train_dict()
        ret = {'optret': opt_res, 'ttd': opt_out}

        save_path = os.path.join(out_path, f'{model_name}_bs_{seed}.pickle')
        with open(save_path, 'wb') as f:
        	pickle.dump(ret, f)

import copy
import torch
import os
import sys

import numpy as np
from auto_LiRPA.bound_general import BoundGeneral

# from gpu_profile import gpu_profile

from config import load_config, get_path, config_modelloader, config_dataloader
from argparser import argparser
from train_general import Train, Logger
from model_defs import model_mlp_after_flatten

# sys.settrace(gpu_profile)


def main(args):
    config = load_config(args)
    global_eval_config = config["eval_params"]
    models, model_names = config_modelloader(config, load_pretrain = False)

    model_ori = model_mlp_after_flatten(in_dim=784, neurons=[64, 64])
    dummy_input = torch.randn(10, 784)
    converted_models = [BoundGeneral(model_ori, dummy_input)]

    robust_errs = []
    errs = []

    checkpoint = torch.load(args.path_prefix, map_location='cpu')
    converted_models[0].load_state_dict(checkpoint['state_dict'], strict=True)

    for model, model_id, model_config in zip(converted_models, model_names, config["models"]):
        model = model.cuda()

        # make a copy of global training config, and update per-model config
        eval_config = copy.deepcopy(global_eval_config)
        if "eval_params" in model_config:
            eval_config.update(model_config["eval_params"])

        # read training parameters from config file
        method = eval_config["method"]
        verbose = eval_config["verbose"]
        eps = eval_config["epsilon"]
        # parameters specific to a training method
        method_param = eval_config["method_params"]
        norm = float(eval_config["norm"])
        train_data, test_data = config_dataloader(config, **eval_config["loader_params"])

        # model_name = get_path(config, model_id, "model", load =False)
        # print(model_name)
        config["path_prefix"] = os.path.split(os.path.split(config["path_prefix"])[0])[0]
        model_log = get_path(config, model_id, "eval_log")
        print(model_log)
        logger = Logger(open(model_log, "w"))
        logger.log("evaluation configurations:", eval_config)
            
        logger.log("Evaluating...")
        with torch.no_grad():
            # evaluate
            print('using bound', eval_config["epsilon_weights"], "norm", norm)
            l2_ball_list = []
            _c = 0
            for i in range(len(model.choices)):
                if hasattr(model.choices[i], 'weight'):
                    l2_norm = torch.norm(model.choices[i].weight.data, p=2)
                    l2_ball_list.append(eval_config["epsilon_weights"][_c] * l2_norm)
                    _c += 1

            print('after times Lp norm of weights',l2_ball_list)
            data = train_data

            print('length of data', len(data)*data.batch_size)
            # robust_err, err = Train(model, 0, test_data, eps, eps, eps, norm, logger, verbose, False, None, method, **method_param)
            robust_err, err = Train(model, 0, data, eps, eps, eps, eval_config['epsilon_weights'], eval_config['epsilon_weights'],
                                    norm, logger, verbose, False, None, method, **method_param)

        robust_errs.append(robust_err)
        errs.append(err)

    print('model robust errors (for robustly trained models, not valid for naturally trained models):')
    print(robust_errs)
    robust_errs = np.array(robust_errs)
    print('min: {:.4f}, max: {:.4f}, median: {:.4f}, mean: {:.4f}'.format(np.min(robust_errs), np.max(robust_errs), np.median(robust_errs), np.mean(robust_errs)))
    print('clean errors for models with min, max and median robust errors')
    i_min = np.argmin(robust_errs)
    i_max = np.argmax(robust_errs)
    i_median = np.argsort(robust_errs)[len(robust_errs) // 2]
    print('for min: {:.4f}, for max: {:.4f}, for median: {:.4f}'.format(errs[i_min], errs[i_max], errs[i_median]))
    print('model clean errors:')
    print(errs)
    print('min: {:.4f}, max: {:.4f}, median: {:.4f}, mean: {:.4f}'.format(np.min(errs), np.max(errs), np.median(errs), np.mean(errs)))


if __name__ == "__main__":
    args = argparser()
    main(args)

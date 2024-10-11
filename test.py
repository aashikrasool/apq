# Code for "APQ: Joint Search for Network Architecture, Pruning and Quantization Policy"
# CVPR 2020
# Tianzhe Wang, Kuan Wang, Han Cai, Ji Lin, Zhijian Liu, Song Han
# {usedtobe, kuanwang, hancai, jilin, zhijian, songhan}@mit.edu

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from utils.latency_predictor import LatencyPredictor
import sys
import copy
import argparse
import os
import json
import torch
from elastic_nn.modules.dynamic_op import DynamicSeparableConv2d, DynamicSeparableQConv2d
from elastic_nn.networks.dynamic_quantized_proxyless import DynamicQuantizedProxylessNASNets
from imagenet_codebase.run_manager import ImagenetRunConfig, RunManager

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--exp_dir', type=str, default='./exps/test')
args, _ = parser.parse_known_args()


if __name__ == '__main__':
    # Initialize predictors
    latency_predictor = LatencyPredictor(type='latency')
    energy_predictor = LatencyPredictor(type='energy')

    # Define the experiment directory and architecture file
    arch_dir = os.path.join(args.exp_dir, 'arch', 'arch.json')  # Assuming the architecture file is 'arch.json'
    assert os.path.exists(arch_dir), f"Architecture file does not exist: {arch_dir}"

    # Load architecture and quantization info from the JSON file
    with open(arch_dir, 'r') as f:
        tmp_lst = json.load(f)
    info, q_info = tmp_lst
    print("Architecture Info:", info)
    print("Quantization Info:", q_info)

    # Latency and energy prediction
    X = LatencyPredictor(type='latency')
    print('Latency: {:.2f}ms'.format(X.predict_lat(dict(info, **q_info))))

    Y = LatencyPredictor(type='energy')
    print('Energy: {:.2f}mJ'.format(Y.predict_lat(dict(info, **q_info))))

    # Define the checkpoint path
    ckpt_path = os.path.join(args.exp_dir, 'checkpoint', 'model_best.pth.tar')
    if os.path.exists(ckpt_path):
        DynamicSeparableConv2d.KERNEL_TRANSFORM_MODE = 1
        DynamicSeparableQConv2d.KERNEL_TRANSFORM_MODE = 1

        # Initialize the dynamic proxyless network
        dynamic_proxyless = DynamicQuantizedProxylessNASNets(
            ks_list=[3, 5, 7],
            expand_ratio_list=[4, 6],
            depth_list=[2, 3, 4],
            base_stage_width='proxyless',
            width_mult_list=1.0,
            dropout_rate=0,
            n_classes=1000
        )

        # Load the initial weights for the dynamic proxyless network
        proxylessnas_init = torch.load(
            './models/imagenet-OFA',  # Path to initial weights
            map_location='cpu'
        )['state_dict']
        dynamic_proxyless.load_weights_from_proxylessnas(proxylessnas_init)

        # Training configuration
        init_lr = 1e-3
        run_config = ImagenetRunConfig(
            test_batch_size=1000,
            image_size=224,
            n_worker=16,
            valid_size=5000,
            dataset='imagenet',
            train_batch_size=256,
            init_lr=init_lr,
            n_epochs=30,
        )

        # Initialize run manager
        run_manager = RunManager('~/tmp', dynamic_proxyless, run_config, init=False)

        # Load the best checkpoint
        proxylessnas_init = torch.load(
            ckpt_path,
            map_location='cpu'
        )['state_dict']
        dynamic_proxyless.load_weights_from_proxylessnas(proxylessnas_init)

        # Set active subnet and quantization policy
        dynamic_proxyless.set_active_subnet(**info)
        dynamic_proxyless.set_quantization_policy(**q_info)

        # Validate the network and print accuracy
        acc = run_manager.validate(is_test=True)
        print('Accuracy: {:.1f}'.format(acc[1]))

    else:
        print(f"Checkpoint file not found: {ckpt_path}")
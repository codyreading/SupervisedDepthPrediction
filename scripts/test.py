#!/usr/bin/python3
# -*- coding: utf-8 -*-
from dp.visualizers import build_visualizer
from dp.datasets.loader import build_loader
from dp.core.solver import Solver
from dp.metircs import build_metrics
from dp.utils.pyt_io import create_summary_writer
from dp.utils.config import load_config, print_config
from dp.metircs.average_meter import AverageMeter
"""
@Time    : 2019-12-24 22:54
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : train.py
"""

import os
import argparse
import logging
import warnings
import sys
import time
import torch
from tqdm import tqdm

warnings.filterwarnings("ignore")
logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

# running in parent dir
os.chdir("..")


parser = argparse.ArgumentParser(description='Training script')
parser.add_argument('-c', '--config', type=str)
parser.add_argument('-r', '--resumed', type=str, default=None, required=False)
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument("--workers", default=8, type=int)

args = parser.parse_args()

if not args.config and not args.resumed:
    logging.error('args --config and --resumed should at least one value available.')
    raise ValueError
is_main_process = True if args.local_rank == 0 else False

solver = Solver()

# read config
if args.resumed:
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    continue_state_object = torch.load(args.resumed,
                                       map_location=torch.device("cpu"))
    config = continue_state_object['config']
    solver.init_from_checkpoint(continue_state_object=continue_state_object)
    if is_main_process:
        snap_dir = args.resumed[:-len(args.resumed.split('/')[-1])]
        if not os.path.exists(snap_dir):
            logging.error('[Error] {} is not existed.'.format(snap_dir))
            raise FileNotFoundError
else:
    config = load_config(args.config)
    solver.init_from_scratch(config)
    if is_main_process:
        exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
        snap_dir = os.path.join(config["snap"]["path"], config['data']['name'],
                                config['model']['name'], exp_time)
        if not os.path.exists(snap_dir):
            os.makedirs(snap_dir)

if is_main_process:
    print_config(config)

te_loader, _, niter_test = build_loader(config,
                                        False,
                                        solver.world_size,
                                        solver.distributed,
                                        num_workers=args.workers)


"""
    usage: debug
"""
# niter_per_epoch, niter_test = 200, 20

loss_meter = AverageMeter()
metric = build_metrics(config)
epoch = 66
if is_main_process:
    writer = create_summary_writer(snap_dir)
    visualizer = build_visualizer(config, writer)
    solver.after_epoch(epoch=epoch)

    # validation
    if is_main_process:
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(niter_test), file=sys.stdout, bar_format=bar_format)
    else:
        pbar = range(niter_test)

    metric.reset()
    test_iter = iter(te_loader)
    for idx in pbar:
        t_start = time.time()
        minibatch = test_iter.next()
        filtered_kwargs = solver.parse_kwargs(minibatch)

        # print(filtered_kwargs)
        t_end = time.time()
        io_time = t_end - t_start
        t_start = time.time()
        pred = solver.step_no_grad(**filtered_kwargs)
        t_end = time.time()
        inf_time = t_end - t_start

        t_start = time.time()
        metric.compute_metric(pred, filtered_kwargs)
        t_end = time.time()
        cmp_time = t_end - t_start

        if is_main_process:
            print_str = '[Test] Epoch{}/{}'.format(epoch, config['solver']['epochs']) \
                        + ' Iter{}/{}: '.format(idx + 1, niter_test) \
                        + metric.get_snapshot_info() \
                        + ' IO:%.2f' % io_time \
                        + ' Inf:%.2f' % inf_time \
                        + ' Cmp:%.2f' % cmp_time
            pbar.set_description(print_str, refresh=False)
        """
        visualization for model output and feature maps.
        """
        if is_main_process and idx % 10 == 0:
            visualizer.visualize(minibatch, pred, epoch=epoch)

    if is_main_process:
        logging.info('After Epoch{}/{}, {}'.format(epoch, config['solver']['epochs'], metric.get_result_info()))
        writer.add_scalar("Train/loss", loss_meter.mean(), epoch)
        metric.add_scalar(writer, tag='Test', epoch=epoch)

if is_main_process:
    writer.close()

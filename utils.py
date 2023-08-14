import logging
import os
import time
import numpy as np
import random
import colorlog
import torch
import yaml

from sklearn.metrics import accuracy_score, f1_score


def random_seed(seed_value):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # multi-GPU
    """Set value for numpy and random"""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed_value)
    random.seed(seed_value)


def accuracy(net, loader, DEVICE):
    """Return accuracy on a dataset given by the data loader."""
    correct = 0
    f1 = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = net(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        f1 += f1_score(targets.cpu().numpy(), predicted.cpu().numpy(), average='micro')*targets.size(0)
    
    return dict({
        "acc": correct / total,
        "f1": f1 / total,
        "len": len(loader.dataset)})



# def record_time(params: Params, t=None, name=None):
#     if t and name and params.save_timing == name or params.save_timing is True:
#         torch.cuda.synchronize()
#         params.timing_data[name].append(round(1000 * (time.perf_counter() - t)))

def create_table(params: dict):
    data = "| name | value | \n |-----|-----|"

    for key, value in params.items():
        data += '\n' + f"| {key} | {value} |"

    return data

def create_logger():
    """
        Setup the logging environment
    """
    log = logging.getLogger()  # root logger
    log.setLevel(logging.DEBUG)
    format_str = '%(asctime)s - %(levelname)-8s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    if os.isatty(2):
        cformat = '%(log_color)s' + format_str
        colors = {'DEBUG': 'reset',
                  'INFO': 'reset',
                  'WARNING': 'bold_yellow',
                  'ERROR': 'bold_red',
                  'CRITICAL': 'bold_red'}
        formatter = colorlog.ColoredFormatter(cformat, date_format,
                                              log_colors=colors)
    else:
        formatter = logging.Formatter(format_str, date_format)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    log.addHandler(stream_handler)
    return logging.getLogger(__name__)


def make_folders(folder_path="./logs"):
    log = create_logger()
    # try:
    #     os.mkdir(folder_path)
    # except FileExistsError:
    #     log.info('Folder already exists')

    # fh = logging.FileHandler(
    #     filename=f'{folder_path}/log.txt')
    # formatter = logging.Formatter('%(asctime)s - %(name)s '
    #                                 '- %(levelname)s - %(message)s')
    # fh.setFormatter(formatter)
    # log.addHandler(fh)

    # with open(f'{folder_path}/params.yaml.txt', 'w') as f:
    #     yaml.dump(self.params, f)
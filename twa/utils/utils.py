import torch
import ruamel.yaml as yaml
import numpy as np
import errno
import os
from datetime import datetime
import torch.nn.functional as F
from sklearn import metrics
import pandas as pd
import glob
import re


def glob_re(pattern, strings):
    return list(filter(re.compile(pattern).match, strings))


def str_to_list(s):
    """
    Converts str of list of floats to a list of floats
    """
    if isinstance(s, str):
        return [float(i) for i in s.split(',')]
    else:
        return s

def strtuple_to_list(param_ranges_tuple):
    """
    Converts tuple of strs of floats to a list of lists of floats
    """
    if isinstance(param_ranges_tuple, tuple):
        param_ranges = [] # TODO: handle input mistakes
        for ir, r in enumerate(param_ranges_tuple):
            vals = r.split(',')
            param_ranges.append([float(vals[0]), float(vals[1])])
        return param_ranges
    else:
        return param_ranges_tuple


def ensure_dir(path: str) -> str:
    """ Creates the directories specified by path if they do not already exist.

    Parameters:
        path (str): path to directory that should be created

    Returns:
        return_path (str): path to the directory that now exists
    """
    path = os.path.abspath(path)
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exception:
            # if the exception is raised because the directory already exits,
            # than our work is done and everything is OK, otherwise re-raise the error
            # THIS CAN OCCUR FROM A POSSIBLE RACE CONDITION!!!
            if exception.errno != errno.EEXIST:
                raise
    return path


def timestamp():
    """
    Return string of current time
    """
    now = datetime.now()
    return now.strftime("%H:%M:%S_%m_%d_%Y")


def nearly_square(n):
    """

    """
    pairs = [(i, int(n / i)) for i in range(1, int(n ** 0.5) + 1) if n % i == 0]
    diffs = [np.abs(a - b) for (a, b) in pairs]
    min_pair = pairs[np.argmin(diffs)]
    if 1 in min_pair:
        q = int(np.ceil(np.sqrt(n)))
        return (q, q)
    else:
        return min_pair

# def stable_sigmoid(x):
#     return torch.where(x < 0, torch.exp(x) / (1 + torch.exp(x)), torch.exp(-1*x) / ((1+torch.exp(-1*x))))

def ensure_device(device: str):
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('cuda is not available')
    return device


################################################ YAML INTERFACE ########################################################


def read_yaml(yaml_file: str) -> dict:
    """ Read a yaml file into dict object

    Parameters:
        yaml_file (str): path to yaml file

    Returns:
        return_dict (dict): dict of yaml contents
    """
    with open(yaml_file, 'r') as f:
        yml = yaml.YAML(typ='safe')
        return yml.load(f)


def write_yaml(yaml_file: str, data: dict) -> None:
    """ Write a dict object into a yaml file

    Parameters:
        yaml_file (str): path to yaml file
        data (dict): dict of data to write to `yaml_file`
    """
    yaml.SafeDumper.ignore_aliases = lambda *args: True

    with open(yaml_file, 'w') as f:
        yaml.safe_dump(data, f)


def update_yaml(yaml_file: str, data: dict) -> None:
    """ Update a yaml file with a dict object

    Parameters:
        yaml_file (str): path to yaml file
        data (dict): dict of data to write to `yaml_file`
    """

    with open(yaml_file, 'r') as f:
        yml = yaml.YAML(typ='safe')
        cur_yaml = yml.load(f)
        cur_yaml.update(data)

    write_yaml(yaml_file, cur_yaml)


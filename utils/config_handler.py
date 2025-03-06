# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

import yaml
from addict import Dict

def read_yaml(file_path, encoding='utf-8'):
    """ read yaml file and convert to easydict

    :param encoding: (str) encoding method uses utf-8 by default
    :return: Dict (addict), the usage of Dict is the same as dict
    """
    with open(file_path, encoding=encoding) as f:
        config = Dict(yaml.load(f.read(), Loader=yaml.FullLoader))
    config.learning_rate = float(config.learning_rate)
    config.warmup_ratio = float(config.warmup_ratio)
    config.eps = float(config.eps)
    return config

def save_yaml(data, out_file_path, encoding='utf-8'):
    """ save dict or easydict to yaml file

    :param data: (dict or Dict(addict)) dict containing the contents of the yaml file
    """
    with open(out_file_path, encoding=encoding, mode='w') as f:
        return yaml.dump(addict2dict(data) if isinstance(data, Dict) else data,
            stream=f,
            allow_unicode=True)

def addict2dict(addict_obj):
    '''convert addict to dict

    :param addict_obj: (Dict(addict)) the addict obj that you want to convert to dict

    :return: (Dict) converted result
    '''
    dict_obj = {}
    for key, vals in addict_obj.items():
        dict_obj[key] = addict2dict(vals) if isinstance(vals, Dict) else vals
    return dict_obj

import os, gzip
import json
import numpy as np
import torch
from tqdm import tqdm
import time
from datasets.build_dataset import build_dataset
import utils.metrics as metrics
from models.build_model import build_model
from utils.arg_parser import Parser
from utils.logger import build_logger


import seaborn as sns

from scipy import ndimage

def resize(input, out_shape):
    dimension = input.shape
    result = ndimage.zoom(input, (out_shape[0] / dimension[0], out_shape[1] / dimension[1]), order=3)
    return result

def build_metric(metric_name):
    return metrics.__dict__[metric_name.lower()]


def test():
    
    argp = Parser()
    arg = argp.parser.parse_args()
    arg_dict = vars(arg)

    torch.set_default_dtype(torch.float64)

    gpu = None
    pretrained = None
    if arg.gpu is not None:
        gpu = int(arg.gpu)
    if arg.pretrained is not None:
        pretrained = arg.pretrained

    if arg.args is not None:
        with open(arg.args, 'rt') as f:
            arg_dict.update(json.load(f))
    arg_dict['test_mode'] = True
    if gpu is not None:
        arg_dict['gpu'] = gpu
    if pretrained is not None:
        arg_dict['pretrained'] = pretrained
    if pretrained is not None and arg_dict['test_mode']:
        arg_dict['save_path'] = os.path.dirname(pretrained)

    logger, log_dir = build_logger(arg_dict)
    logger.info(arg_dict)

        
    if arg_dict['cpu']:
        device = torch.device("cpu")
        logger.info('using cpu for training')
    elif arg_dict['gpu'] is not None:
        torch.cuda.set_device(arg_dict['gpu'])
        device = torch.device("cuda", arg_dict['gpu'])
        logger.info('using gpu {} for training'.format(arg_dict['gpu']))
        
    logger.info('===> Loading datasets')
    # Initialize dataset
    dataset = build_dataset(arg_dict)
    logger.info('===> Building model')

    
    # Initialize model parameters
    model = build_model(arg_dict)

    if not arg_dict['cpu']:
        model = model.to(device)
    count = 1
    for feature, instance_count_path, instance_name_path in dataset:
        design_name = os.path.basename(instance_count_path[0])
        if 'FPU' in design_name:
            design_name = 'RISCY-FPU'
        else:
            design_name = design_name.split('_')[0]

        if arg_dict['cpu']:
            input = feature
            start_time = time.time()
            prediction = model(input)
            end_time = time.time()
        else:
            input = feature.to(device)
            torch.cuda.synchronize()
            start_time = time.time()
            prediction = model(input)
            torch.cuda.synchronize()
            end_time = time.time()
        logger.info('#{} {}, inference time {}s'.format(count, os.path.basename(instance_count_path[0][:-4]), end_time - start_time))

  
        instance_count = np.load(instance_count_path[0].replace('instance_count', 'instance_count_from_power_rpt')).astype(int)
        instance_name = np.load(instance_name_path[0].replace('instance_name', 'instance_name_from_power_rpt'))['instance_name'] # load npz
        pred_irdrop = resize(prediction[0,0,:,:].detach().cpu().numpy(), instance_count.shape)
        pred_instance_irdrop = np.repeat(pred_irdrop.ravel(),instance_count.ravel())
 
        # 输出预测的static_ir report
        # 文件为2列，第一列是vdd_drop+gnd_bounce，第二列是inst_name，不需要表头。
        # 文件名为pred_static_ir_{case name}(.gz)。建议按下面的方式以gzip形式输出，文件名加上.gz。若不压缩则不需要.gz。
        file_name = os.path.splitext(os.path.basename(instance_count_path[0]))[0]
        output_path = '/home/edalab/Desktop/ml_irdrop/out_1128'
        
        if not os.path.exists(os.path.join(output_path, 'pred_static_ir_report')):
            os.makedirs(os.path.join(output_path, 'pred_static_ir_report'))

        with gzip.open('{}/{}/{}'.format(output_path, 'pred_static_ir_report', 'pred_static_ir_{}.gz'.format(file_name)), 'wt') as f:
            for i,j in zip(pred_instance_irdrop, instance_name):
                f.write('{} {}\n'.format(i,j))
        count +=1

if __name__ == "__main__":
    test()

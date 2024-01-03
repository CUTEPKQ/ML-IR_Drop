import os
import argparse
import numpy as np
from scipy import ndimage
from multiprocessing import Process

dir_path = os.path.dirname(os.path.abspath(__file__))


def get_sub_path(path):
    sub_path = []
    if isinstance(path, list):
        for p in path:
            if os.path.isdir(p):
                for file in os.listdir(p):
                    sub_path.append(os.path.join(p, file))
            else:
                continue
    else:
        for file in os.listdir(path):
            sub_path.append(os.path.join(path, file))
    return sub_path

def std(input):
    if input.max() == 0:
        return input
    else:
        result = (input-input.min()) / (input.max()-input.min())
        return result
        
def resize(input):
    dimension = input.shape
    result = ndimage.zoom(input, (256 / dimension[0], 256 / dimension[1]), order=3)
    return result

def save_npy(out_list, save_path, name):
    output = np.array(out_list)
    output = np.transpose(output, (1, 2, 0))
    np.save(os.path.join(save_path, name), output)

def divide_n(list_in, n):
    list_out = [ [] for i in range(n)]
    for i,e in enumerate(list_in):
        list_out[i%n].append(e)
    return list_out

def pack_data(args, name_list, read_feature_list, save_path):
    feature_save_path = os.path.join(args.save_path, args.prefix, 'feature')
    for name in name_list:
        out_feature_list = []
        for feature_name in read_feature_list:
            name = os.path.basename(name)
            feature = np.load(os.path.join(args.data_path, feature_name, name))
            feature = std(resize(feature))
            out_feature_list.append(feature)
        save_npy(out_feature_list, feature_save_path, name)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default = '1128', type=str, help = 'dir name')
    parser.add_argument("--data_path", default = os.path.join(dir_path,'../..','out_1128'), type=str, help = 'path to the decompressed dataset')
    parser.add_argument("--save_path",  default = os.path.join(dir_path,'dataset'), type=str, help = 'path to save training set')
    parser.add_argument('--process_capacity', default=29, help='number of process for multi process')

    args = parser.parse_args()                                       
    return args

if __name__ == '__main__':
    args = parse_args()

    feature_list = ['features/total_power', 'features/eff_res_VDD', 'features/eff_res_VSS']
    name_list = get_sub_path(os.path.join(args.data_path, feature_list[0]))
    print('processing %s files' % len(name_list))
    save_path = os.path.join(args.save_path, args.prefix)
    feature_save_path = os.path.join(args.save_path, args.prefix, 'feature')

    if not os.path.exists(feature_save_path):
        os.makedirs(feature_save_path)


    nlist = divide_n(name_list, args.process_capacity)
    process = []
    for l in nlist:
        p = Process(target=pack_data, args=(args, l, feature_list, save_path))
        process.append(p)
    for p in process:
        p.start()
    for p in process:
        p.join()


    






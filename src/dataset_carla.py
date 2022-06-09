import os
import pickle
import zlib
import multiprocessing
from multiprocessing import Process

import numpy as np
import torch
from tqdm import tqdm

import utils
from utils import get_name
from carla_with_traffic import get_vectornet_mapping


class Dataset(torch.utils.data.Dataset):
    def __init__(self, args, batch_size, to_screen=True):
        data_dir = args.data_dir
        self.ex_list = []
        self.args = args

        if args.reuse_temp_file:
            pickle_file = open(os.path.join(args.temp_file_dir, get_name('ex_list')), 'rb')
            self.ex_list = pickle.load(pickle_file)
            # self.ex_list = self.ex_list[len(self.ex_list) // 2:]
            pickle_file.close()
        else:
            files = []
            for each_dir in data_dir:
                root, dirs, cur_files = os.walk(each_dir).__next__()
                files.extend([os.path.join(each_dir, file) for file in cur_files if
                                file.startswith('vehicles_pos_list_')])
            print(files[:5], files[-5:])
            pbar = tqdm(total=len(files))

            args.core_num = min(args.core_num, len(files)//2)
            queue = multiprocessing.Queue(args.core_num)
            queue_res = multiprocessing.Queue()

            def calc_ex_list(file_path, queue, queue_res, args):
                res = []
                dis_list = []
                while True:
                    file = queue.get()
                    if file is None:
                        break
                    if file.endswith("npy"):
                        num_str_start_index = len(file_path)+18
                        agent_angle_block_path = file_path+'agent_angle_'+file[num_str_start_index:]
                        start_idx = int(file[num_str_start_index:-4])
                        # print(start_idx)
                        print("start processing:" + str(agent_angle_block_path))
                        vehicles_pos_lists_block = np.load(file)
                        agent_angle_block = np.load(agent_angle_block_path)
                        bound_info = np.load(file_path+'bound_info.npy', allow_pickle=True).item()
                        # print(file_path+'bound_info.npy')
                        lane_info = np.load(file_path+'lane_info.npy', allow_pickle=True).item()
                        for i in range(1000):
                            instance = {}
                            vehicles_pos_list, angle = vehicles_pos_lists_block[i], agent_angle_block[i]
                            get_vectornet_mapping(vehicles_pos_list, angle, bound_info, lane_info, start_idx+i, instance)
                            if instance is not None:
                                data_compress = zlib.compress(pickle.dumps(instance))
                                res.append(data_compress)
                                queue_res.put(data_compress)
                            else:
                                queue_res.put(None)
                        print("Done:" + str(agent_angle_block_path))

            processes = [Process(target=calc_ex_list, args=(data_dir[0], queue, queue_res, args,)) for _ in range(args.core_num)]
            for each in processes:
                each.start()
            # res = pool.map_async(calc_ex_list, [queue for i in range(args.core_num)])
            for file in files:
                assert file is not None
                queue.put(file)
                pbar.update(1)

            # necessary because queue is out-of-order
            while not queue.empty():
                pass

            pbar.close()

            self.ex_list = []

            pbar = tqdm(total=len(files)*1000)
            for i in range(len(files)*1000):
                t = queue_res.get()
                if t is not None:
                    self.ex_list.append(t)
                pbar.update(1)
            pbar.close()
            pass

            for i in range(args.core_num):
                queue.put(None)
            for each in processes:
                each.join()

            pickle_file = open(os.path.join(args.temp_file_dir, get_name('ex_list')), 'wb')
            pickle.dump(self.ex_list, pickle_file)
            pickle_file.close()
        assert len(self.ex_list) > 0
        if to_screen:
            print("valid data size is", len(self.ex_list))
            # logging('max_vector_num', max_vector_num)
        self.batch_size = batch_size

    def __len__(self):
        return len(self.ex_list)

    def __getitem__(self, idx):
        # file = self.ex_list[idx]
        # pickle_file = open(file, 'rb')
        # instance = pickle.load(pickle_file)
        # pickle_file.close()

        data_compress = self.ex_list[idx]
        instance = pickle.loads(zlib.decompress(data_compress))
        return instance


def post_eval(args, file2pred, file2labels, DEs):
    from argoverse.evaluation import eval_forecasting

    score_file = args.model_recover_path.split('/')[-1]
    for each in args.eval_params:
        each = str(each)
        if len(each) > 15:
            each = 'long'
        score_file += '.' + str(each)
        # if 'minFDE' in args.other_params:
        #     score_file += '.minFDE'
    if args.method_span[0] >= utils.NMS_START:
        score_file += '.NMS'
    else:
        score_file += '.score'

    for method in utils.method2FDEs:
        FDEs = utils.method2FDEs[method]
        miss_rate = np.sum(np.array(FDEs) > 2.0) / len(FDEs)
        if method >= utils.NMS_START:
            method = 'NMS=' + str(utils.NMS_LIST[method - utils.NMS_START])
        utils.logging(
            'method {}, FDE {}, MR {}, other_errors {}'.format(method, np.mean(FDEs), miss_rate, utils.other_errors_to_string()),
            type=score_file, to_screen=True, append_time=True)
    utils.logging('other_errors {}'.format(utils.other_errors_to_string()),
                  type=score_file, to_screen=True, append_time=True)
    metric_results = eval_forecasting.get_displacement_errors_and_miss_rate(file2pred, file2labels, 6, 30, 2.0)
    utils.logging(metric_results, type=score_file, to_screen=True, append_time=True)
    DE = np.concatenate(DEs, axis=0)
    length = DE.shape[1]
    DE_score = [0, 0, 0, 0]
    for i in range(DE.shape[0]):
        DE_score[0] += DE[i].mean()
        for j in range(1, 4):
            index = round(float(length) * j / 3) - 1
            assert index >= 0
            DE_score[j] += DE[i][index]
    for j in range(4):
        score = DE_score[j] / DE.shape[0]
        utils.logging('ADE' if j == 0 else 'DE@1' if j == 1 else 'DE@2' if j == 2 else 'DE@3', score,
                      type=score_file, to_screen=True, append_time=True)

    utils.logging(vars(args), is_json=True,
                  type=score_file, to_screen=True, append_time=True)

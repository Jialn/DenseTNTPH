'''
Run the eval realtime with carla.

Run example:
python3 src/run_eval_realtime.py --argoverse --future_frame_num 30 \
  --output_dir models.densetnt.1 --hidden_size 128 --train_batch_size 64 --use_map \
  --core_num 16 --use_centerline --distributed_training 1 \
  --other_params \
    semantic_lane direction goals_2D enhance_global_graph subdivide lazy_points laneGCN point_sub_graph \
    stage_one stage_one_dynamic=0.95 laneGCN-4 point_level-4-3 complete_traj \
    set_predict=6 set_predict-6 data_ratio_per_epoch=0.4 set_predict-topk=0 set_predict-one_encoder set_predict-MRratio=1.0 \
    set_predict-train_recover=models.densetnt.set_predict.1/model_save/model.16.bin --do_eval \
    --data_dir_for_val /media/jiangtao.li/simu_machine_dat/argoverse/val_200/data/ --reuse_temp_file # --visualize

'''
import argparse
import logging
import os
from functools import partial

import numpy as np
import torch
from torch.utils.data import SequentialSampler

import structs
import utils
from modeling.vectornet import VectorNet

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def eval_instance_argoverse(batch_size, args, pred, mapping, file2pred, file2labels, DEs):
    for i in range(batch_size):
        a_pred = pred[i]
        assert a_pred.shape == (6, args.future_frame_num, 2)
        file_name_int = int(os.path.split(mapping[i]['file_name'])[1][:-4])
        file2pred[file_name_int] = a_pred
        if not args.do_test:
            file2labels[file_name_int] = mapping[i]['origin_labels']

    if not args.do_test:
        DE = np.zeros([batch_size, args.future_frame_num])
        for i in range(batch_size):
            origin_labels = mapping[i]['origin_labels']
            for j in range(args.future_frame_num):
                DE[i][j] = np.sqrt((origin_labels[j][0] - pred[i, 0, j, 0]) ** 2 + (
                        origin_labels[j][1] - pred[i, 0, j, 1]) ** 2)
        DEs.append(DE)
        miss_rate = 0.0
        if 0 in utils.method2FDEs:
            FDEs = utils.method2FDEs[0]
            miss_rate = np.sum(np.array(FDEs) > 2.0) / len(FDEs)



def do_eval(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    print("Loading Evalute Dataset", args.data_dir)
    if args.argoverse:
        from dataset_argoverse import Dataset
    eval_dataset = Dataset(args, args.eval_batch_size)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.eval_batch_size,
                                                  sampler=eval_sampler,
                                                  collate_fn=utils.batch_list_to_batch_tensors,
                                                  pin_memory=False)
    model = VectorNet(args)
    print('torch.cuda.device_count', torch.cuda.device_count())

    logger.info("***** Recover model: %s *****", args.model_recover_path)
    if args.model_recover_path is None:
        raise ValueError("model_recover_path not specified.")

    model_recover = torch.load(args.model_recover_path)
    model.load_state_dict(model_recover, strict=False)

    if 'set_predict-train_recover' in args.other_params and 'complete_traj' in args.other_params:
        model_recover = torch.load(args.other_params['set_predict-train_recover'])
        if 'do_train' in args.other_params:
            utils.load_model(model.decoder.complete_traj_cross_attention, model_recover, prefix='decoder.complete_traj_cross_attention.')
            utils.load_model(model.decoder.complete_traj_decoder, model_recover, prefix='decoder.complete_traj_decoder.')
        else:
            utils.load_model(model.decoder.set_predict_decoders, model_recover, prefix='decoder.set_predict_decoders.')
            utils.load_model(model.decoder.set_predict_encoders, model_recover, prefix='decoder.set_predict_encoders.')
            utils.load_model(model.decoder.set_predict_point_feature, model_recover, prefix='decoder.set_predict_point_feature.')

    model.to(device)
    model.eval()
    file2pred = {}
    file2labels = {}
    DEs = []

    argo_pred = structs.ArgoPred()

    for batch in eval_dataloader:
        pred_trajectory, pred_score, _ = model(batch, device)

        mapping = batch
        batch_size = pred_trajectory.shape[0]
        print(batch_size)
        for i in range(batch_size):
            assert pred_trajectory[i].shape == (6, args.future_frame_num, 2)
            assert pred_score[i].shape == (6,)
            argo_pred[mapping[i]['file_name']] = structs.MultiScoredTrajectory(pred_score[i].copy(), pred_trajectory[i].copy())
        print(mapping[0].keys())
        if args.argoverse:
            eval_instance_argoverse(batch_size, args, pred_trajectory, mapping, file2pred, file2labels, DEs)
        break

    if args.argoverse:
        from dataset_argoverse import post_eval
        post_eval(args, file2pred, file2labels, DEs)

'''
mapping
dict_keys([
    'file_name', 'start_time', 'city_name', 'cent_x', 'cent_y', '
    agent_pred_index', 'two_seconds', 'origin_labels', 'angle', 'trajs', 'agents', 'map_start_polyline_idx', 'polygons', 
    'goals_2D', 'goals_2D_labels', 'stage_one_label', 'matrix', 'labels', 'polyline_spans', 'labels_is_valid', 'eval_time', 
    'stage_one_scores', 'stage_one_topk', 'set_predict_ans_points', 'vis.predict_trajs'])

'''



def main():
    parser = argparse.ArgumentParser()
    utils.add_argument(parser)
    args: utils.Args = parser.parse_args()
    utils.init(args, logger)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logger.info("device: {}".format(device))

    do_eval(args)
    logger.info('Finish.')

if __name__ == "__main__":
    main()


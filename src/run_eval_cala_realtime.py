'''
Run the eval realtime with carla.

Run example:
python3 src/run_eval_cala_realtime.py --argoverse --future_frame_num 30 \
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
from cv2 import imshow

import numpy as np
import torch
from torch.utils.data import SequentialSampler

import structs
import utils
from modeling.vectornet import VectorNet
from carla_with_traffic import CarlaSyncModeWithTraffic, draw_matrix


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
carla_client = CarlaSyncModeWithTraffic()

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

    # load set_predictor model
    model_recover = torch.load(args.other_params['set_predict-train_recover'])
    utils.load_model(model.decoder.set_predict_decoders, model_recover, prefix='decoder.set_predict_decoders.')
    utils.load_model(model.decoder.set_predict_encoders, model_recover, prefix='decoder.set_predict_encoders.')
    utils.load_model(model.decoder.set_predict_point_feature, model_recover, prefix='decoder.set_predict_point_feature.')

    model.to(device)
    model.eval()
    file2pred = {}
    file2labels = {}
    DEs = []

    test_mapping = None
    # only load the first batch for testing
    for batch in eval_dataloader:
        test_mapping = batch
        break
    print(test_mapping[-1].keys())

    for i in range(64):
        carla_client.tick()
        # print(test_mapping[-1]['matrix'])  # 239
        print("length of original matrix:")
        print(len(test_mapping[i]['matrix']))  # 239
        print("map start idx:")
        print(test_mapping[i]['map_start_polyline_idx'])
        print("polyline_spans:")
        print(test_mapping[i]['polyline_spans'])

        # use the last one
        test_mapping[-1]['matrix'], test_mapping[-1]['polyline_spans'], test_mapping[-1]['map_start_polyline_idx'], \
            test_mapping[-1]['trajs'] = carla_client.get_vectornet_input()

        # run the model
        pred_trajectory, pred_score, _ = model(test_mapping, device)

        # visulaize
        draw_matrix(test_mapping[i]['matrix'], test_mapping[i]['polyline_spans'], test_mapping[i]['map_start_polyline_idx'], 
            pred_trajectory=test_mapping[i]['vis.predict_trajs'], win_name="argoverse", wait_key=10)
        draw_matrix(test_mapping[-1]['matrix'], test_mapping[-1]['polyline_spans'], test_mapping[-1]['map_start_polyline_idx'], 
            pred_trajectory=test_mapping[-1]['vis.predict_trajs'], wait_key=10)
        batch_size = pred_trajectory.shape[0]
        for i in range(batch_size):
            assert pred_trajectory[i].shape == (6, args.future_frame_num, 2)
            assert pred_score[i].shape == (6,)

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
    try:
        main()
    finally:
        carla_client.destroy_vechicles()


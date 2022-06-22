'''
Run the eval realtime with carla.

To run using optimization, set 'set_predict_recover' to None, then run with:
python3 src/_scripts/run_eval_cala_realtime.py --argoverse --future_frame_num 30 \
  --output_dir models/models.densetnt.1 --hidden_size 128 --eval_batch_size 1 --use_map \
  --core_num 16 --use_centerline --distributed_training 1 \
  --other_params \
    semantic_lane direction goals_2D enhance_global_graph subdivide lazy_points laneGCN point_sub_graph \
    stage_one stage_one_dynamic=0.95 laneGCN-4 point_level-4-3 complete_traj \
    --do_eval --eval_params optimization MRminFDE cnt_sample=9 opti_time=0.1

To run using set-predictor, set 'set_predict_recover' to the model path, then run with:
python3 src/carla_scripts/run_eval_cala_realtime.py --argoverse --future_frame_num 30 \
  --output_dir models/models.densetnt.1 --hidden_size 128 --eval_batch_size 1 --use_map \
  --core_num 16 --use_centerline --distributed_training 1 \
  --other_params \
    semantic_lane direction goals_2D enhance_global_graph subdivide lazy_points laneGCN point_sub_graph \
    stage_one stage_one_dynamic=0.95 laneGCN-4 point_level-4-3 complete_traj \
    set_predict=6 set_predict-6 data_ratio_per_epoch=0.4 set_predict-topk=0 set_predict-one_encoder set_predict-MRratio=1.0 \
    --do_eval #--reuse_temp_file #--visualize

Model path is passed by 'output_dir', but can be overrid by compete_traj_recover and set_predict_recover in this file
'''
import os, sys
import argparse
import logging
import numpy as np
import torch
from torch.utils.data import SequentialSampler

sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/..')
import utils
from modeling.vectornet import VectorNet
from carla_with_traffic import CarlaSyncModeWithTraffic, draw_vectornet_mapping

run_realtime_on_carla = True  # for testing on carla
max_testing_steps = 2000
# for offline testing on argoverse or carla dataset, only for testing purpose
run_offline_testing = False
data_dir_for_offline_testing = 'carla_offline_data/16000/'  # ..../argoverse/val_200/data/
# extra model recover path to override
set_predict_recover = 'models/train_with_argoverse.set_predict.epoch16.bin'  # if set to None, optimizer should be used in args
compete_traj_recover = None  # 'models/argoverse_finetune_with_100K_carla.epoch6.bin'


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

if run_realtime_on_carla:
    carla_client = CarlaSyncModeWithTraffic()


def eval_instance_carla(batch_size, args, pred, mapping, file2pred, file2labels, DEs):
    for i in range(batch_size):
        a_pred = pred[i]
        assert a_pred.shape == (6, args.future_frame_num, 2)
        file_name_int = int(mapping[i]['file_name'][6:])
        file2pred[file_name_int] = a_pred
        if not args.do_test:
            file2labels[file_name_int] = mapping[i]['origin_labels']
    if not args.do_test:
        DE = np.zeros([batch_size, args.future_frame_num])
        for i in range(batch_size):
            origin_labels = mapping[i]['origin_labels']
            for j in range(args.future_frame_num):
                # pred[batch_size, 6, x, y]:  why [i, 0, j, 0] [i, 0, j, 1]? 
                DE[i][j] = np.sqrt((origin_labels[j][0] - pred[i, 0, j, 0]) ** 2 + (
                        origin_labels[j][1] - pred[i, 0, j, 1]) ** 2)
        DEs.append(DE)


def post_eval(args, file2pred, file2labels, DEs):
    from argoverse.evaluation import eval_forecasting
    metric_results = eval_forecasting.get_displacement_errors_and_miss_rate(file2pred, file2labels, 6, 30, 2.0)
    utils.logging(metric_results, to_screen=True, append_time=True)
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
        utils.logging('ADE' if j == 0 else 'DE@1' if j == 1 else 'DE@2' if j == 2 else 'DE@3', score, to_screen=True, append_time=True)
    utils.logging(vars(args), is_json=True, to_screen=True, append_time=True)


def do_eval(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    model = VectorNet(args)
    print('torch.cuda.device_count', torch.cuda.device_count())
    logger.info("***** Recover model: %s *****", args.model_recover_path)
    if args.model_recover_path is None:
        raise ValueError("model_recover_path not specified.")
    model_recover = torch.load(args.model_recover_path)
    model.load_state_dict(model_recover, strict=False)
    # load extra model
    if set_predict_recover is not None and 'set_predict' in args.other_params:
        model_recover = torch.load(set_predict_recover)
        utils.load_model(model.decoder.set_predict_decoders, model_recover, prefix='decoder.set_predict_decoders.')
        utils.load_model(model.decoder.set_predict_encoders, model_recover, prefix='decoder.set_predict_encoders.')
        utils.load_model(model.decoder.set_predict_point_feature, model_recover, prefix='decoder.set_predict_point_feature.')
    if compete_traj_recover is not None:
        model_recover = torch.load(compete_traj_recover)
        utils.load_model(model.decoder.goals_2D_mlps, model_recover, prefix='decoder.goals_2D_mlps.')
        utils.load_model(model.decoder.complete_traj_cross_attention, model_recover, prefix='decoder.complete_traj_cross_attention.')
        utils.load_model(model.decoder.complete_traj_decoder, model_recover, prefix='decoder.complete_traj_decoder.')
        # print(model.state_dict()['decoder.complete_traj_cross_attention.value.weight'])
        
    model.to(device)
    model.eval()

    if run_offline_testing:
        args.data_dir = [data_dir_for_offline_testing]
        print("Loading Evalute Dataset", args.data_dir)
        if os.path.exists(args.data_dir[0]+'lane_info.npy'):
            from carla.dataset_carla import Dataset
        else: # 'carla'
            from argoverse_scripts.dataset_argoverse import Dataset
        eval_dataset = Dataset(args, args.eval_batch_size)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.eval_batch_size,
                                                    sampler=eval_sampler,
                                                    collate_fn=utils.batch_list_to_batch_tensors,
                                                    pin_memory=False)
        argoverse_batch = []
        for batch in eval_dataloader:
            argoverse_batch.append(batch)
        print("argoverse_batch length:")
        print(len(argoverse_batch))
    
    test_mapping = [{}]*args.eval_batch_size
    import structs
    carla_pred = structs.ArgoPred()
    file2pred = {}
    file2labels = {}
    DEs = []
    loop_cnt = 0
    while True:
        if run_realtime_on_carla:
            # get input from carla
            carla_client.tick()
            carla_client.get_vectornet_input(test_mapping[0])
            # run the model
            pred_trajectory, pred_score, _ = model(test_mapping, device)
            # visulaize
            draw_vectornet_mapping(test_mapping[0], wait_key=10, win_name='carla_vis') # wait_key=10 or None
            print("length of original matrix:" + str(len(test_mapping[0]['matrix'])))
            print("map start idx:" + str(test_mapping[0]['map_start_polyline_idx']))
            # print("polyline_spans:" + str(test_mapping['polyline_spans']))
            batch_size = pred_trajectory.shape[0]
            for i in range(batch_size):
                assert pred_trajectory[i].shape == (6, args.future_frame_num, 2)
                assert pred_score[i].shape == (6,)
                carla_pred[test_mapping[i]['file_name']] = structs.MultiScoredTrajectory(pred_score[i].copy(), pred_trajectory[i].copy())
                eval_instance_carla(batch_size, args, pred_trajectory, test_mapping, file2pred, file2labels, DEs)

        if run_offline_testing:
            batch = argoverse_batch[loop_cnt%len(argoverse_batch)]
            pred_trajectory, pred_score, _ = model(batch, device)
            draw_vectornet_mapping(batch[0], wait_key=None, win_name='offline_eval_vis')
            batch[0].clear()

        loop_cnt += 1
        print("loop_cnt: " + str(loop_cnt))
        if loop_cnt > max_testing_steps: break
    post_eval(args, file2pred, file2labels, DEs)

def main():
    parser = argparse.ArgumentParser()
    utils.add_argument(parser)
    args: utils.Args = parser.parse_args()
    utils.init(args, logger)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logger.info("device: {}".format(device))
    try:
        do_eval(args)
    finally:
        if 'optimization' in args.other_params:
            utils.select_goals_by_optimization(None, None, close=True)
        if run_realtime_on_carla:
            carla_client.destroy_vechicles()


if __name__ == "__main__":
    main()


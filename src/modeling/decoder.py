"""
Outline of decoder:
decoder(mapping, batch_size, lane_states_batch, inputs, inputs_lengths, hidden_states, device),定义在Decoder.forward(mapping: List[Dict], batch_size, lane_states_batch: List[Tensor], inputs: Tensor,inputs_lengths: List[int], hidden_states: Tensor, device)函数:
    初始化保存loss相关的变量
    从mapping中提取goals_2D,goals_2D是从地图车道数据中构建的采样点坐标 - 代表车道周边区域热力图
    调用goals_2D_per_example(i, goals_2D, mapping, lane_states_batch, inputs, inputs_lengths, hidden_states, labels, labels_is_valid, device, loss, DE)
        从mapping[labels]获取gt_points,即30个轨迹点的xy坐标
        调用goals_2D_per_example_stage_one(i, mapping, lane_states_batch, ..., loss)
            计算StageOne得分:hidden_states(即global_graph的输出)经过CrossAttention和一个MLP decoder得到stage_one_scores,是一个list,长度和map里面的lane的数量一致,实际上代表车道线的得分.
            用stage_one_label计算loss(第一个loss),stage_one_label是所有车道中心线里面距离轨迹末端点最近的那个车道线的index
            筛选得分较高的车道,阈值stage_one_dynamic,默认0.95,动态的stage_one topk数量,表示只统计前95%得分的车道
            返回stage_one_topk_ids,表示第一阶段车道线评分高的几个车道index
        get_scores(goals_2D_tensor, *get_scores_inputs)获取goals2D热力图的得分,stage_one是考虑车道线得分,这里是goals2D热力图,两个有差别
            先经过PointSubGraph(goals_2D经过3层MLP,后2层会concat进来前面global_graph的编码hidden_states）得到goals_2D_hidden
            goals_2D_hidden和inputs经过一个CrossAttention,得到goals_2D_hidden_attention
            goals_2D_hidden和stage_one_topk也经过一个CrossAttention,得到stage_one_goals_2D_hidden_attention
            goals_2D_hidden,stage_one_topk和stage_one_goals_2D_hidden_attention经过stage_one_goals_2D_decoder然后送进socres解码器stage_one_goals_2D_decoder
                这个decoder跟前面的stage_one_decoder类似,一个MLP decoder
            解码得到的scores经过softmax然后取log后返回
        goals_2D_per_example_lazy_points() 综合考虑goals2D的valuemap得分和车道线得分,获取scores, highest_goal, goals_2D可视化结果
            这个函数通过utils.get_neighbour_points获取得分较高的vlauemap里面的点周边的点
            然后concat起来新的点和原来的点,再送进去get_scores计算一次
            备注：不清楚作用,论文里面似乎没讲这个,直观理解是对得分较高的区域附近点进行扩充,让其超过原来车道附近,有更大的区域
        如果在train,使用goals_2D_per_example_calc_loss函数计算loss,stage_one_label的loss前面已经计算,这里主要是轨迹生成loss和得分loss（第2,3个loss）
            F.smooth_l1_loss(predict_traj, torch.tensor(gt_points, dtype=torch.float, device=device), reduction='none')
            F.nll_loss(scores.unsqueeze(0), torch.tensor([mapping[i]['goals_2D_labels']], device=device)
        保存可视化的相关变量
        如果set_predict,使用神经网络根据goals_2D和scores选择goal,否则使用优化方法select_goals_by_NMS。
        run_set_predict(goals_2D, scores, mapping, device, loss, i):
            即替代优化的方法,用神经网络去预测goalset
            把xy坐标构成的goals_2D加上一个得分维度,得到vectors_3D(x,y,score),然后送进去2层3*128,128*128的MLP得到points_feature
            然后points_feature送进去k个set-predictor的encoder和decoder,encoder就是GlobalGraphRes,decoder就是DecoderResCat
            每个decoder都会precit出6个xy坐标decoding[1:].view([6, 2])和得分decoding[0],总共有k个
            如果在do train,这里还有第四个loss: loss[i] += 2.0 * F.l1_loss(predicts[min_cost_idx], torch.tensor(dynamic_label, device=device, dtype=torch.float))
            这里的dynamic_label通过utils_cython.set_predict_next_step计算得来,有一个参数set_predict-MRratio控制优化MissRate还是优化ADE/FDE
            最后根据得分和一些其他规则筛选goal,保存到mapping[i]['set_predict_ans_points']
    if do_eval: goals_2D_eval
        根据上面输出的结果选择出goal:pred_goals_batch = [mapping[i]['set_predict_ans_points']
        complete_traj,即根据goal完成trajectory:
            根据goal生成轨迹:goals_2D_mlps complete_traj_cross_attention complete_traj_decoder得到轨迹predict_trajs
            对predict_trajs经过坐标变换,由待预测车辆的第一人称相对坐标转换回绝对坐标,用于后续评估
    if visualize: 调用visualize_goals_2D,保存可视化的图片

"""

from typing import Dict, List, Tuple, NamedTuple, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor

import structs
import utils_cython
from modeling.lib import PointSubGraph, GlobalGraphRes, CrossAttention, GlobalGraph, MLP

import utils


class DecoderRes(nn.Module):
    def __init__(self, hidden_size, out_features=60):
        super(DecoderRes, self).__init__()
        self.mlp = MLP(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, out_features)

    def forward(self, hidden_states):
        hidden_states = hidden_states + self.mlp(hidden_states)
        hidden_states = self.fc(hidden_states)
        return hidden_states


class DecoderResCat(nn.Module):
    def __init__(self, hidden_size, in_features, out_features=60):
        super(DecoderResCat, self).__init__()
        self.mlp = MLP(in_features, hidden_size)
        self.fc = nn.Linear(hidden_size + in_features, out_features)

    def forward(self, hidden_states):
        hidden_states = torch.cat([hidden_states, self.mlp(hidden_states)], dim=-1)
        hidden_states = self.fc(hidden_states)
        return hidden_states


class Decoder(nn.Module):

    def __init__(self, args_: utils.Args, vectornet):
        super(Decoder, self).__init__()
        global args
        args = args_
        hidden_size = args.hidden_size
        self.future_frame_num = args.future_frame_num
        self.mode_num = args.mode_num

        self.decoder = DecoderRes(hidden_size, out_features=2)

        if 'variety_loss' in args.other_params:
            self.variety_loss_decoder = DecoderResCat(hidden_size, hidden_size, out_features=6 * self.future_frame_num * 2)

            if 'variety_loss-prob' in args.other_params:
                self.variety_loss_decoder = DecoderResCat(hidden_size, hidden_size, out_features=6 * self.future_frame_num * 2 + 6)
        elif 'goals_2D' in args.other_params:
            # self.decoder = DecoderResCat(hidden_size, hidden_size, out_features=self.future_frame_num * 2)
            self.goals_2D_mlps = nn.Sequential(
                MLP(2, hidden_size),
                MLP(hidden_size),
                MLP(hidden_size)
            )
            # self.goals_2D_decoder = DecoderRes(hidden_size * 3, out_features=1)
            self.goals_2D_decoder = DecoderResCat(hidden_size, hidden_size * 3, out_features=1)
            self.goals_2D_cross_attention = CrossAttention(hidden_size)
            if 'point_sub_graph' in args.other_params:
                self.goals_2D_point_sub_graph = PointSubGraph(hidden_size)
        if 'stage_one' in args.other_params:
            self.stage_one_cross_attention = CrossAttention(hidden_size)
            self.stage_one_decoder = DecoderResCat(hidden_size, hidden_size * 3, out_features=1)
            self.stage_one_goals_2D_decoder = DecoderResCat(hidden_size, hidden_size * 4, out_features=1)

        if 'set_predict' in args.other_params:
            if args.do_train:
                if 'set_predict-train_recover' in args.other_params:
                    model_recover = torch.load(args.other_params['set_predict-train_recover'])
                else:
                    model_recover = torch.load(args.model_recover_path)
                vectornet.decoder = self
                utils.load_model(vectornet, model_recover)
                # self must be vectornet
                for p in vectornet.parameters():
                    p.requires_grad = False

            self.set_predict_point_feature = nn.Sequential(MLP(3, hidden_size), MLP(hidden_size, hidden_size))

            self.set_predict_encoders = nn.ModuleList(
                [GlobalGraphRes(hidden_size) for _ in range(args.other_params['set_predict'])])

            self.set_predict_decoders = nn.ModuleList(
                [DecoderResCat(hidden_size, hidden_size * 2, out_features=13) for _ in range(args.other_params['set_predict'])])

    def goals_2D_per_example_stage_one(self, i, mapping, lane_states_batch, inputs, inputs_lengths,
                                       hidden_states, device, loss):
        def get_stage_one_scores():
            stage_one_hidden = lane_states_batch[i]
            stage_one_hidden_attention = self.stage_one_cross_attention(
                stage_one_hidden.unsqueeze(0), inputs[i][:inputs_lengths[i]].unsqueeze(0)).squeeze(0)
            stage_one_scores = self.stage_one_decoder(torch.cat([hidden_states[i, 0, :].unsqueeze(0).expand(
                stage_one_hidden.shape), stage_one_hidden, stage_one_hidden_attention], dim=-1))
            stage_one_scores = stage_one_scores.squeeze(-1)
            stage_one_scores = F.log_softmax(stage_one_scores, dim=-1)
            return stage_one_scores

        stage_one_scores = get_stage_one_scores()
        assert len(stage_one_scores) == len(mapping[i]['polygons'])
        mapping[i]['stage_one_scores'] = stage_one_scores
        # print('stage_one_scores', stage_one_scores.requires_grad)
        loss[i] += F.nll_loss(stage_one_scores.unsqueeze(0),
                              torch.tensor([mapping[i]['stage_one_label']], device=device))
        # print('stage_one_scores-2', loss[i].requires_grad)
        if 'stage_one_dynamic' in args.other_params:
            _, stage_one_topk_ids = torch.topk(stage_one_scores, k=len(stage_one_scores))
            threshold = float(args.other_params['stage_one_dynamic'])
            sum = 0.0
            for idx, each in enumerate(torch.exp(stage_one_scores[stage_one_topk_ids])):
                sum += each
                if sum > threshold:
                    stage_one_topk_ids = stage_one_topk_ids[:idx + 1]
                    break
            utils.other_errors_put('stage_one_k', len(stage_one_topk_ids))
        else:
            _, stage_one_topk_ids = torch.topk(stage_one_scores, k=min(args.stage_one_K, len(stage_one_scores)))

        if mapping[i]['stage_one_label'] in stage_one_topk_ids.tolist():
            utils.other_errors_put('stage_one_recall', 1.0)
        else:
            utils.other_errors_put('stage_one_recall', 0.0)

        stage_one_topk = lane_states_batch[i][stage_one_topk_ids]
        mapping[i]['stage_one_topk'] = stage_one_topk

        return stage_one_topk_ids

    def goals_2D_per_example_lazy_points(self, i, goals_2D, mapping, labels, device, scores,
                                         get_scores_inputs, stage_one_topk_ids=None, gt_points=None):
        if args.argoverse:
            k = 150
        else:
            k = 40
        _, topk_ids = torch.topk(scores, k=min(k, len(scores)))
        topk_ids = topk_ids.tolist()

        goals_2D_new = utils.get_neighbour_points(goals_2D[topk_ids], topk_ids=topk_ids, mapping=mapping[i])

        goals_2D_new = torch.cat([torch.tensor(goals_2D_new, device=device, dtype=torch.float),
                                  torch.tensor(goals_2D, device=device, dtype=torch.float)], dim=0)

        old_vector_num = len(goals_2D)

        goals_2D = np.array(goals_2D_new.tolist())
        # print('len', len(goals_2D))

        scores = self.get_scores(goals_2D_new, *get_scores_inputs)

        index = torch.argmax(scores).item()
        point = np.array(goals_2D_new[index].tolist())

        if not args.do_test:
            label = np.array(labels[i]).reshape([self.future_frame_num, 2])
            final_idx = mapping[i].get('final_idx', -1)
            mapping[i]['goals_2D_labels'] = np.argmin(utils.get_dis(goals_2D, label[final_idx]))

        return scores, point, goals_2D

    def goals_2D_per_example_calc_loss(self, i: int, goals_2D: np.ndarray, mapping: List[Dict], inputs: Tensor,
                                       inputs_lengths: List[int], hidden_states: Tensor, device, loss: Tensor,
                                       DE: np.ndarray, gt_points: np.ndarray, scores: Tensor, highest_goal: np.ndarray,
                                       labels_is_valid: List[np.ndarray]):
        """
        Calculate loss for a training example
        """
        final_idx = mapping[i].get('final_idx', -1)
        DE[i][final_idx] = np.sqrt((highest_goal[0] - gt_points[final_idx][0]) ** 2 + (highest_goal[1] - gt_points[final_idx][1]) ** 2)
        if 'complete_traj' in args.other_params:
            target_feature = self.goals_2D_mlps(torch.tensor(gt_points[final_idx], dtype=torch.float, device=device))
            pass
            if True:
                target_feature.detach_()
                hidden_attention = self.complete_traj_cross_attention(
                    target_feature.unsqueeze(0).unsqueeze(0), inputs[i][:inputs_lengths[i]].detach().unsqueeze(0)).squeeze(
                    0).squeeze(0)
                predict_traj = self.complete_traj_decoder(
                    torch.cat([hidden_states[i, 0, :].detach(), target_feature, hidden_attention], dim=-1)).view(
                    [self.future_frame_num, 2])
            loss[i] += (F.smooth_l1_loss(predict_traj, torch.tensor(gt_points, dtype=torch.float, device=device), reduction='none') * \
                        torch.tensor(labels_is_valid[i], dtype=torch.float, device=device).view(self.future_frame_num, 1)).mean()

        loss[i] += F.nll_loss(scores.unsqueeze(0),
                              torch.tensor([mapping[i]['goals_2D_labels']], device=device))

    def goals_2D_per_example(self, i: int, goals_2D: np.ndarray, mapping: List[Dict], lane_states_batch: List[Tensor],
                             inputs: Tensor, inputs_lengths: List[int], hidden_states: Tensor, labels: List[np.ndarray],
                             labels_is_valid: List[np.ndarray], device, loss: Tensor, DE: np.ndarray):
        """
        :param i: example index in batch
        :param goals_2D: candidate goals sampled from map (shape ['goal num', 2])
        :param lane_states_batch: each value in list is hidden states of lanes (value shape ['lane num', hidden_size])
        :param inputs: hidden states of all elements before encoding by global graph (shape [batch_size, 'element num', hidden_size])
        :param inputs_lengths: valid element number of each example
        :param hidden_states: hidden states of all elements after encoding by global graph (shape [batch_size, -1, hidden_size])
        :param loss: (shape [batch_size])
        :param DE: displacement error (shape [batch_size, self.future_frame_num])
        """
        if args.do_train:
            final_idx = mapping[i].get('final_idx', -1)
            assert labels_is_valid[i][final_idx]

        gt_points = labels[i].reshape([self.future_frame_num, 2])

        stage_one_topk_ids = None
        if 'stage_one' in args.other_params: # MarkJT
            stage_one_topk_ids = self.goals_2D_per_example_stage_one(i, mapping, lane_states_batch, inputs, inputs_lengths,
                                                                     hidden_states, device, loss)

        goals_2D_tensor = torch.tensor(goals_2D, device=device, dtype=torch.float)
        get_scores_inputs = (inputs, hidden_states, inputs_lengths, i, mapping, device)

        scores = self.get_scores(goals_2D_tensor, *get_scores_inputs)
        index = torch.argmax(scores).item()
        highest_goal = goals_2D[index]

        if 'lazy_points' in args.other_params: # MarkJT
            scores, highest_goal, goals_2D = \
                self.goals_2D_per_example_lazy_points(i, goals_2D, mapping, labels, device, scores,
                                                      get_scores_inputs, stage_one_topk_ids, gt_points)
            index = None

        if args.do_train:
            self.goals_2D_per_example_calc_loss(i, goals_2D, mapping, inputs, inputs_lengths,
                                                hidden_states, device, loss, DE, gt_points, scores, highest_goal, labels_is_valid)

        mapping[i]['vis.goals_2D'] = goals_2D
        mapping[i]['vis.scores'] = np.array(scores.tolist())
        if args.visualize:
            # mapping[i]['vis.scores'] = np.array(scores.tolist())
            mapping[i]['vis.labels'] = gt_points
            mapping[i]['vis.labels_is_valid'] = labels_is_valid[i]

        if 'set_predict' in args.other_params:
            self.run_set_predict(goals_2D, scores, mapping, device, loss, i)
            if args.visualize:
                set_predict_ans_points = mapping[i]['set_predict_ans_points']
                predict_trajs = np.zeros((6, self.future_frame_num, 2))
                predict_trajs[:, -1, :] = set_predict_ans_points

        else:
            if args.do_eval:
                if args.nms_threshold is not None:
                    utils.select_goals_by_NMS(mapping[i], goals_2D, np.array(scores.tolist()), args.nms_threshold, mapping[i]['speed'])
                elif 'optimization' in args.other_params:
                    mapping[i]['goals_2D_scores'] = goals_2D.astype(np.float32), np.array(scores.tolist(), dtype=np.float32)
                else:
                    assert False

    def goals_2D_eval(self, batch_size, mapping, labels, hidden_states, inputs, inputs_lengths, device):
        if 'set_predict' in args.other_params:      # MarkJT
            pred_goals_batch = [mapping[i]['set_predict_ans_points'] for i in range(batch_size)]
            pred_probs_batch = np.zeros((batch_size, 6))
        elif 'optimization' in args.other_params:   # MarkJT
            pred_goals_batch, pred_probs_batch = utils.select_goals_by_optimization(
                np.array(labels).reshape([batch_size, self.future_frame_num, 2]), mapping)
        elif args.nms_threshold is not None:
            pred_goals_batch = [mapping[i]['pred_goals'] for i in range(batch_size)]
            pred_probs_batch = [mapping[i]['pred_probs'] for i in range(batch_size)]
        else:
            assert False

        pred_goals_batch = np.array(pred_goals_batch)
        pred_probs_batch = np.array(pred_probs_batch)
        assert pred_goals_batch.shape == (batch_size, self.mode_num, 2)
        assert pred_probs_batch.shape == (batch_size, self.mode_num)

        if 'complete_traj' in args.other_params:    # MarkJT
            pred_trajs_batch = []
            for i in range(batch_size):
                targets_feature = self.goals_2D_mlps(torch.tensor(pred_goals_batch[i], dtype=torch.float, device=device))
                hidden_attention = self.complete_traj_cross_attention(
                    targets_feature.unsqueeze(0), inputs[i][:inputs_lengths[i]].unsqueeze(0)).squeeze(0)
                predict_trajs = self.complete_traj_decoder(
                    torch.cat([hidden_states[i, 0, :].unsqueeze(0).expand(len(targets_feature), -1), targets_feature,
                               hidden_attention], dim=-1)).view([self.mode_num, self.future_frame_num, 2])
                predict_trajs = np.array(predict_trajs.tolist())
                final_idx = mapping[i].get('final_idx', -1)
                predict_trajs[:, final_idx, :] = pred_goals_batch[i]
                mapping[i]['vis.predict_trajs'] = predict_trajs.copy()

                if args.argoverse:
                    for each in predict_trajs:
                        utils.to_origin_coordinate(each, i)
                pred_trajs_batch.append(predict_trajs)
            pred_trajs_batch = np.array(pred_trajs_batch)
        else:
            pass
        if args.visualize:
            for i in range(batch_size):
                utils.visualize_goals_2D(mapping[i], mapping[i]['vis.goals_2D'], mapping[i]['vis.scores'], self.future_frame_num,
                                         labels=mapping[i]['vis.labels'],
                                         labels_is_valid=mapping[i]['vis.labels_is_valid'],
                                         predict=mapping[i]['vis.predict_trajs'])

        return pred_trajs_batch, pred_probs_batch, None

    def variety_loss(self, mapping: List[Dict], hidden_states: Tensor, batch_size, inputs: Tensor,
                     inputs_lengths: List[int], labels_is_valid: List[np.ndarray], loss: Tensor,
                     DE: np.ndarray, device, labels: List[np.ndarray]):
        """
        :param hidden_states: hidden states of all elements after encoding by global graph (shape [batch_size, -1, hidden_size])
        :param inputs: hidden states of all elements before encoding by global graph (shape [batch_size, 'element num', hidden_size])
        :param inputs_lengths: valid element number of each example
        :param DE: displacement error (shape [batch_size, self.future_frame_num])
        """
        outputs = self.variety_loss_decoder(hidden_states[:, 0, :])
        pred_probs = None
        if 'variety_loss-prob' in args.other_params:
            pred_probs = F.log_softmax(outputs[:, -6:], dim=-1)
            outputs = outputs[:, :-6].view([batch_size, 6, self.future_frame_num, 2])
        else:
            outputs = outputs.view([batch_size, 6, self.future_frame_num, 2])

        for i in range(batch_size):
            if args.do_train:
                assert labels_is_valid[i][-1]
            gt_points = np.array(labels[i]).reshape([self.future_frame_num, 2])
            argmin = np.argmin(utils.get_dis_point_2_points(gt_points[-1], np.array(outputs[i, :, -1, :].tolist())))

            loss_ = F.smooth_l1_loss(outputs[i, argmin],
                                     torch.tensor(gt_points, device=device, dtype=torch.float), reduction='none')
            loss_ = loss_ * torch.tensor(labels_is_valid[i], device=device, dtype=torch.float).view(self.future_frame_num, 1)
            if labels_is_valid[i].sum() > utils.eps:
                loss[i] += loss_.sum() / labels_is_valid[i].sum()

            if 'variety_loss-prob' in args.other_params:
                loss[i] += F.nll_loss(pred_probs[i].unsqueeze(0), torch.tensor([argmin], device=device))
        if args.do_eval:
            outputs = np.array(outputs.tolist())
            pred_probs = np.array(pred_probs.tolist(), dtype=np.float32) if pred_probs is not None else pred_probs
            for i in range(batch_size):
                for each in outputs[i]:
                    utils.to_origin_coordinate(each, i)

            return outputs, pred_probs, None
        return loss.mean(), DE, None

    def forward(self, mapping: List[Dict], batch_size, lane_states_batch: List[Tensor], inputs: Tensor,
                inputs_lengths: List[int], hidden_states: Tensor, device):
        """
        :param lane_states_batch: each value in list is hidden states of lanes (value shape ['lane num', hidden_size])
        :param inputs: hidden states of all elements before encoding by global graph (shape [batch_size, 'element num', hidden_size])
        :param inputs_lengths: valid element number of each example
        :param hidden_states: hidden states of all elements after encoding by global graph (shape [batch_size, 'element num', hidden_size])
        """
        labels = utils.get_from_mapping(mapping, 'labels')
        labels_is_valid = utils.get_from_mapping(mapping, 'labels_is_valid')
        loss = torch.zeros(batch_size, device=device)
        DE = np.zeros([batch_size, self.future_frame_num])

        if 'variety_loss' in args.other_params:
            return self.variety_loss(mapping, hidden_states, batch_size, inputs, inputs_lengths, labels_is_valid, loss, DE, device, labels)
        elif 'goals_2D' in args.other_params:
            for i in range(batch_size):
                goals_2D = mapping[i]['goals_2D']

                self.goals_2D_per_example(i, goals_2D, mapping, lane_states_batch, inputs, inputs_lengths,
                                          hidden_states, labels, labels_is_valid, device, loss, DE)


            if args.do_eval:
                return self.goals_2D_eval(batch_size, mapping, labels, hidden_states, inputs, inputs_lengths, device)
            else:
                if args.visualize:
                    for i in range(batch_size):
                        predict = np.zeros((self.mode_num, self.future_frame_num, 2))
                        utils.visualize_goals_2D(mapping[i], mapping[i]['vis.goals_2D'], mapping[i]['vis.scores'],
                                                 self.future_frame_num,
                                                 labels=mapping[i]['vis.labels'],
                                                 labels_is_valid=mapping[i]['vis.labels_is_valid'],
                                                 predict=predict)
                return loss.mean(), DE, None
        else:
            assert False

    def get_scores(self, goals_2D_tensor: Tensor, inputs, hidden_states, inputs_lengths, i, mapping, device):
        """
        :param goals_2D_tensor: candidate goals sampled from map (shape ['goal num', 2])
        :return: log scores of goals (shape ['goal num'])
        """
        if 'point_sub_graph' in args.other_params:
            goals_2D_hidden = self.goals_2D_point_sub_graph(goals_2D_tensor.unsqueeze(0), hidden_states[i, 0:1, :]).squeeze(0)
        else:
            goals_2D_hidden = self.goals_2D_mlps(goals_2D_tensor)

        goals_2D_hidden_attention = self.goals_2D_cross_attention(
            goals_2D_hidden.unsqueeze(0), inputs[i][:inputs_lengths[i]].unsqueeze(0)).squeeze(0)

        if 'stage_one' in args.other_params:
            stage_one_topk = mapping[i]['stage_one_topk']
            stage_one_scores = mapping[i]['stage_one_scores']
            stage_one_topk_here = stage_one_topk
            stage_one_goals_2D_hidden_attention = self.goals_2D_cross_attention(
                goals_2D_hidden.unsqueeze(0), stage_one_topk_here.unsqueeze(0)).squeeze(0)
            li = [hidden_states[i, 0, :].unsqueeze(0).expand(goals_2D_hidden.shape),
                  goals_2D_hidden, goals_2D_hidden_attention, stage_one_goals_2D_hidden_attention]

            scores = self.stage_one_goals_2D_decoder(torch.cat(li, dim=-1))
        else:
            scores = self.goals_2D_decoder(torch.cat([hidden_states[i, 0, :].unsqueeze(0).expand(
                goals_2D_hidden.shape), goals_2D_hidden, goals_2D_hidden_attention], dim=-1))

        scores = scores.squeeze(-1)
        scores = F.log_softmax(scores, dim=-1)
        return scores

    def run_set_predict(self, goals_2D, scores, mapping, device, loss, i):
        gt_points = mapping[i]['labels'].reshape((self.future_frame_num, 2))

        if args.argoverse:
            if 'set_predict-topk' in args.other_params:
                topk_num = args.other_params['set_predict-topk']

                if topk_num == 0:
                    topk_num = torch.sum(scores > np.log(0.00001)).item()

                _, topk_ids = torch.topk(scores, k=min(topk_num, len(scores)))
                goals_2D = goals_2D[topk_ids.cpu().numpy()]
                scores = scores[topk_ids]

        scores_positive_np = np.exp(np.array(scores.tolist(), dtype=np.float32))
        goals_2D = goals_2D.astype(np.float32)

        max_point_idx = torch.argmax(scores)
        vectors_3D = torch.cat([torch.tensor(goals_2D, device=device, dtype=torch.float), scores.unsqueeze(1)], dim=-1)
        vectors_3D = torch.tensor(vectors_3D.tolist(), device=device, dtype=torch.float)

        vectors_3D[:, 0] -= goals_2D[max_point_idx, 0]
        vectors_3D[:, 1] -= goals_2D[max_point_idx, 1]

        points_feature = self.set_predict_point_feature(vectors_3D)
        costs = np.zeros(args.other_params['set_predict'])
        pseudo_labels = []
        predicts = []

        group_scores = torch.zeros([len(self.set_predict_encoders)], device=device)

        if True:
            for k, (encoder, decoder) in enumerate(zip(self.set_predict_encoders, self.set_predict_decoders)):
                if 'set_predict-one_encoder' in args.other_params:
                    encoder = self.set_predict_encoders[0]

                if True:
                    if 'set_predict-one_encoder' in args.other_params and k > 0:
                        pass
                    else:
                        encoding = encoder(points_feature.unsqueeze(0)).squeeze(0)

                    decoding = decoder(torch.cat([torch.max(encoding, dim=0)[0], torch.mean(encoding, dim=0)], dim=-1)).view([13])
                    group_scores[k] = decoding[0]
                    predict = decoding[1:].view([6, 2])

                    predict[:, 0] += goals_2D[max_point_idx, 0]
                    predict[:, 1] += goals_2D[max_point_idx, 1]

                predicts.append(predict)

                if args.do_eval:
                    pass
                else:
                    selected_points = np.array(predict.tolist(), dtype=np.float32)
                    temp = None
                    assert goals_2D.dtype == np.float32, goals_2D.dtype
                    kwargs = None
                    if 'set_predict-MRratio' in args.other_params:
                        kwargs = {}
                        kwargs['set_predict-MRratio'] = args.other_params['set_predict-MRratio']
                    costs[k] = utils_cython.set_predict_get_value(goals_2D, scores_positive_np, selected_points, kwargs=kwargs)

                    pseudo_labels.append(temp)

        argmin = torch.argmax(group_scores).item()

        if args.do_train:
            utils.other_errors_put('set_hungary', np.min(costs))
            group_scores = F.log_softmax(group_scores, dim=-1)
            min_cost_idx = np.argmin(costs)
            loss[i] = 0

            if True:
                selected_points = np.array(predicts[min_cost_idx].tolist(), dtype=np.float32)
                kwargs = None
                if 'set_predict-MRratio' in args.other_params:
                    kwargs = {}
                    kwargs['set_predict-MRratio'] = args.other_params['set_predict-MRratio']
                _, dynamic_label = utils_cython.set_predict_next_step(goals_2D, scores_positive_np, selected_points,
                                                                      lr=args.set_predict_lr, kwargs=kwargs)
                # loss[i] += 2.0 / globals.set_predict_lr * \
                #            F.l1_loss(predicts[min_cost_idx], torch.tensor(dynamic_label, device=device, dtype=torch.float))
                loss[i] += 2.0 * F.l1_loss(predicts[min_cost_idx], torch.tensor(dynamic_label, device=device, dtype=torch.float))

            loss[i] += F.nll_loss(group_scores.unsqueeze(0), torch.tensor([min_cost_idx], device=device))

            t = np.array(predicts[min_cost_idx].tolist())

            utils.other_errors_put('set_MR_mincost', np.min(utils.get_dis_point_2_points(gt_points[-1], t)) > 2.0)
            utils.other_errors_put('set_minFDE_mincost', np.min(utils.get_dis_point_2_points(gt_points[-1], t)))

        predict = np.array(predicts[argmin].tolist())

        set_predict_ans_points = predict.copy()
        li = []
        for point in set_predict_ans_points:
            li.append((point, scores[np.argmin(utils.get_dis_point_2_points(point, goals_2D))]))
        li = sorted(li, key=lambda x: -x[1])
        set_predict_ans_points = np.array([each[0] for each in li])
        mapping[i]['set_predict_ans_points'] = set_predict_ans_points

        if args.argoverse:
            utils.other_errors_put('set_MR_pred', np.min(utils.get_dis_point_2_points(gt_points[-1], predict)) > 2.0)
            utils.other_errors_put('set_minFDE_pred', np.min(utils.get_dis_point_2_points(gt_points[-1], predict)))

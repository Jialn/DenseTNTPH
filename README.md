# DenseTNT
### [Paper](https://arxiv.org/abs/2108.09640) | [Webpage](https://tsinghua-mars-lab.github.io/DenseTNT/)
- This is the official implementation of the paper: **DenseTNT: End-to-end Trajectory Prediction from Dense Goal Sets** (ICCV 2021).
- **DenseTNT v1.0** was released in November 1st, 2021.

## Quick Start

Requires:

* Python ≥ 3.6
* PyTorch ≥ 1.6

### 1) Install Packages

``` bash
 pip install -r requirements.txt
```

### 2) Install Argoverse API
The latest version of Argoverse requires Python ≥ 3.7

If using Python 3.6, you can install Argoverse v1.0 

https://github.com/argoai/argoverse-api

### 3) Compile Cython
Compile a .pyx file into a C file using Cython (already installed at step 1):


⚠️*Recompiling is needed every time the pyx files are changed.*
``` bash
cd src/ && cython -a utils_cython.pyx && python setup.py build_ext --inplace && cd ../
```

## Performance

Results on Argoverse motion forecasting validation set:

<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky"></th>
    <th class="tg-c3ow">minADE</th>
    <th class="tg-c3ow">minFDE</th>
    <th class="tg-c3ow">Miss Rate</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0pky">DenseTNT w/ 100ms optimization (Miss Rate)</td>
    <td class="tg-c3ow">0.80</td>
    <td class="tg-c3ow">1.27</td>
    <td class="tg-c3ow">7.0%</td>
  </tr>
  <tr>
    <td class="tg-0pky">DenseTNT w/ 100ms optimization (minFDE)</td>
    <td class="tg-c3ow">0.73</td>
    <td class="tg-c3ow">1.05</td>
    <td class="tg-c3ow">9.8%</td>
  </tr>
  <tr>
    <td class="tg-0pky">DenseTNT w/ goal set predictor (Miss Rate)</td>
    <td class="tg-c3ow">0.82</td>
    <td class="tg-c3ow">1.37</td>
    <td class="tg-c3ow">7.0%</td>
  </tr>
  <tr>
    <td class="tg-0pky">DenseTNT w/ goal set predictor (minFDE)</td>
    <td class="tg-c3ow">0.75</td>
    <td class="tg-c3ow">1.05</td>
    <td class="tg-c3ow">9.7%</td>
  </tr>
</tbody>
</table>

## DenseTNT

### 1) Train
Suppose the training data of Argoverse motion forecasting is at ```./train/data/```.
```bash
OUTPUT_DIR=models.densetnt.1; \
GPU_NUM=8; \
python3 src/run.py --argoverse --future_frame_num 30 \
  --do_train --data_dir ./train/data/ --output_dir ${OUTPUT_DIR} \
  --hidden_size 128 --train_batch_size 64 --use_map \
  --core_num 16 --use_centerline --distributed_training ${GPU_NUM} \
  --other_params \
    semantic_lane direction  \
    goals_2D enhance_global_graph subdivide lazy_points laneGCN point_sub_graph \
    stage_one stage_one_dynamic=0.95 laneGCN-4 point_level-4-3 complete_traj complete_traj-3
```
Example on my machine:
```
OUTPUT_DIR=models.densetnt.1
python3 src/run.py --argoverse --future_frame_num 30   --do_train --data_dir /media/jiangtao.li/simu_machine_dat/argoverse/train/data --output_dir models.densetnt.1 --hidden_size 128 --train_batch_size 64 --use_map   --core_num 16 --use_centerline --distributed_training 1 --other_params semantic_lane direction goals_2D enhance_global_graph subdivide lazy_points laneGCN point_sub_graph     stage_one stage_one_dynamic=0.95 laneGCN-4 point_level-4-3 complete_traj complete_traj-3 # --reuse_temp_file
```
Training takes 20 minutes per epoch and 5 hours for the default 16 epochs on 8 × 2080Ti. 
Add --reuse_temp_file  to skip re-listing the map file for the second time running

### 2) Evaluate
Suppose the validation data of Argoverse motion forecasting is at ```./val/data/```.

* Optimize Miss Rate:
  - Add ```--do_eval --eval_params optimization MRminFDE cnt_sample=9 opti_time=0.1``` to the end of the training command.

* Optimize minFDE: 
  - Add ```--do_eval --eval_params optimization MRminFDE=0.0 cnt_sample=9 opti_time=0.1``` to the end of the training command.

Example on my machine:
```
python3 src/run.py --argoverse --future_frame_num 30 --output_dir models.densetnt.1   --hidden_size 128 --train_batch_size 64 --use_map   --core_num 16 --use_centerline --distributed_training 1  --other_params     semantic_lane direction      goals_2D enhance_global_graph subdivide lazy_points laneGCN point_sub_graph     stage_one stage_one_dynamic=0.95 laneGCN-4 point_level-4-3 complete_traj complete_traj-3 --do_eval --eval_params optimization MRminFDE cnt_sample=9 opti_time=0.1 --data_dir_for_val /media/jiangtao.li/simu_machine_dat/argoverse/val_200/data/ # --reuse_temp_file --visualize
```
Result:
```
method 0, FDE 1.3026716672555985, MR 0.07202573976489664, other_errors {'stage_one_k': 3.005421564653425, 'stage_one_recall': 0.9601743007701662}
other_errors {'stage_one_k': 3.005421564653425, 'stage_one_recall': 0.9601743007701662}
{'minADE': 0.8216933611058539, 'minFDE': 1.302671667255584, 'MR': 0.07202573976489664}
ADE 1.4395600034558007
DE@1 0.8137086290007429
DE@2 1.781262619181053
DE@3 3.1312567902911526
```

### 3) Train Set Predictor (Optional)
Compared with the optimization algorithm (default setting), the set predictor has similar performance but faster inference speed.


After training DenseTNT, suppose the model path is at ```models.densetnt.1/model_save/model.16.bin```. The command for training the set predictor is:
```bash
OUTPUT_DIR=models.densetnt.set_predict.1; \
MODEL_PATH=models.densetnt.1/model_save/model.16.bin; \
GPU_NUM=8; \
python src/run.py --argoverse --future_frame_num 30 \
  --do_train --data_dir train/data/ --output_dir ${OUTPUT_DIR} \
  --hidden_size 128 --train_batch_size 64 --use_map \
  --core_num 16 --use_centerline --distributed_training ${GPU_NUM} \
  --other_params \
    semantic_lane direction goals_2D enhance_global_graph subdivide lazy_points laneGCN point_sub_graph \
    stage_one stage_one_dynamic=0.95 laneGCN-4 point_level-4-3 complete_traj \
    set_predict=6 set_predict-6 data_ratio_per_epoch=0.4 set_predict-topk=0 set_predict-one_encoder set_predict-MRratio=1.0 \
    set_predict-train_recover=${MODEL_PATH} \
```
Example on my machine:
```bash
python3 src/run.py --argoverse --future_frame_num 30 \
  --do_train --data_dir /media/jiangtao.li/simu_machine_dat/argoverse/train/data/ --output_dir models.densetnt.set_predict.1 \
  --hidden_size 128 --train_batch_size 64 --use_map \
  --core_num 16 --use_centerline --distributed_training 1 \
  --other_params \
    semantic_lane direction goals_2D enhance_global_graph subdivide lazy_points laneGCN point_sub_graph \
    stage_one stage_one_dynamic=0.95 laneGCN-4 point_level-4-3 complete_traj \
    set_predict=6 set_predict-6 data_ratio_per_epoch=0.4 set_predict-topk=0 set_predict-one_encoder set_predict-MRratio=1.0 \
    set_predict-train_recover=models.densetnt.1/model_save/model.16.bin  # --reuse_temp_file
```

This training command optimizes Miss Rate. To optimize minFDE, change ```set_predict-MRratio=1.0``` in the command to ```set_predict-MRratio=0.0```.

To evaluate the set predictor, just add ```--do_eval``` to the end of this training command.
Default eval ADE is very large. finally solved by:
```bash
python3 src/run.py --argoverse --future_frame_num 30 \
  --output_dir models.densetnt.1 --hidden_size 128 --train_batch_size 64 --use_map \
  --core_num 16 --use_centerline --distributed_training 1 \
  --other_params \
    semantic_lane direction goals_2D enhance_global_graph subdivide lazy_points laneGCN point_sub_graph \
    stage_one stage_one_dynamic=0.95 laneGCN-4 point_level-4-3 complete_traj \
    set_predict=6 set_predict-6 data_ratio_per_epoch=0.4 set_predict-topk=0 set_predict-one_encoder set_predict-MRratio=1.0 \
    set_predict-train_recover=models.densetnt.set_predict.1/model_save/model.16.bin --do_eval \
    --data_dir_for_val /media/jiangtao.li/simu_machine_dat/argoverse/val_200/data/ --reuse_temp_file # --visualize
```
and changed codes of the model loading part accrodingly. Result:
```
other_errors {'stage_one_k': 3.005421564653425, 'stage_one_recall': 0.9601743007701662, 'set_MR_pred': 0.07126570733684637, 'set_minFDE_pred': 1.390715484339638}
{'minADE': 0.8472929869140687, 'minFDE': 1.390715484339639, 'MR': 0.07126570733684637}
ADE 1.524842857083471
DE@1 0.8426495707258723
DE@2 1.8900086805419107
DE@3 3.3966121944761225
```

## Citation
If you find our work useful for your research, please consider citing the paper:
```
@inproceedings{densetnt,
  title={Densetnt: End-to-end trajectory prediction from dense goal sets},
  author={Gu, Junru and Sun, Chen and Zhao, Hang},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={15303--15312},
  year={2021}
}
```
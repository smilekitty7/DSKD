
https://mmdetection.readthedocs.io/en/latest/1_exist_data_model.html
https://mmdetection.readthedocs.io/en/latest/3_exist_data_new_model.html


# 模型训练

## 简单分布式训练
tools/dist_train.sh configs/agg_rcnn/agg_r50_fpn_hlkt.py 4 --work-dir=$expdir/common_exp
nohup ./tools/dist_train.sh configs/yolox/yolox_resnet_qoqo.py 4  --work-dir=$expdir/yolox_r50_qoqo3w1k_stst/  1>$expdir/yolox_r50_qoqo3w1k_stst/nohup.out 2>&1 &
nohup ./tools/dist_train.sh configs/deformable_detr/deformdetr_resnet_qoqo.py 4  --work-dir=$expdir/defdetr_mini/  1>$expdir/defdetr_mini/nohup 2>&1 &

### 指定GPU训练
CUDA_VISIBLE_DEVICES=2,3 nohup ./tools/dist_train.sh configs/yolof/yolof_r50_c5_8x8_1x_hlkt.py 4  --work-dir=$expdir/mmd-yolof-hlkt/  1>$expdir/mmd-yolof-hlkt/nohup.out 2>&1 &


### 指定GPU 指定中间点文件
CUDA_VISIBLE_DEVICES=2,3 nohup ./tools/dist_train.sh configs/paa/paa_r50_fpn_1x_hlkt.py 2 --work-dir /home/xdata/zhangjp/experiments/mmd-paa-hlkt/r50-fpn-1x-8x2-800px/ --resume-from /home/xdata/zhangjp/experiments/mmd-paa-hlkt/r50-fpn-1x-8x2-800px/epoch_9.pth  1>/home/xdata/zhangjp/experiments/mmd-paa-hlkt/r50-fpn-1x-8x2-800px/nohup.out 2>&1 &

### 使用软链接路径进行训练
nohup ./tools/dist_train.sh configs/sparse_rcnn/sparse_rcnn_r50_fpn_mstrain_480-640_2x_hlkt.py 4 --work-dir=$expdir/sparse_r50_fpn_mstrain_480-640_2x_hlkt 1>$expdir/sparse_r50_fpn_mstrain_480-640_2x_hlkt/nohup.out 2>&1 &


## 非分布式训练
python tools/train.py configs/balloon/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_balloon.py

python ./tools/train.py configs/xyz_rcnn/xyz_rcnn_r50_xpn_hlkt.py --gpus=4 --work-dir=$expdir/xyz_r50_xpn_hlkt
usage: train.py [-h] [--work-dir WORK_DIR] [--resume-from RESUME_FROM]
                [--no-validate]
                [--gpus GPUS | --gpu-ids GPU_IDS [GPU_IDS ...]] [--seed SEED]
                [--deterministic] [--options OPTIONS [OPTIONS ...]]
                [--cfg-options CFG_OPTIONS [CFG_OPTIONS ...]]
                [--launcher {none,pytorch,slurm,mpi}]
                [--local_rank LOCAL_RANK]

# 模型测试
https://mmdetection.readthedocs.io/en/latest/1_exist_data_model.html?highlight=multi-gpu%20testing

## single-gpu testing
python tools/test.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    [--out ${RESULT_FILE}] \
    [--eval ${EVAL_METRICS}] \
    [--show]
usage: test.py [-h] [--work-dir WORK_DIR] [--out OUT] [--fuse-conv-bn]
               [--format-only] [--eval EVAL [EVAL ...]] [--show]
               [--show-dir SHOW_DIR] [--show-score-thr SHOW_SCORE_THR]
               [--gpu-collect] [--tmpdir TMPDIR]
               [--cfg-options CFG_OPTIONS [CFG_OPTIONS ...]]
               [--options OPTIONS [OPTIONS ...]]
               [--eval-options EVAL_OPTIONS [EVAL_OPTIONS ...]]
               [--launcher {none,pytorch,slurm,mpi}] [--local_rank LOCAL_RANK]
python tools/test.py configs/xxx/xxxx.py $expdir/x/latest.pth --eval=bbox --show=False

## multi-gpu testing
bash tools/dist_test.sh \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    ${GPU_NUM} \
    [--out ${RESULT_FILE}] \
    [--eval ${EVAL_METRICS}] \
    [--show] \ [--show-dir] \ [--show-score-thr]
tools/dist_test.sh configs/xxx/xxxx.py $expdir/x/latest.pth 4 --eval=bbox
nohup tools/dist_test.sh configs/sparse_rcnn/sparse_r50_fpn_qoqo.py $expdir/sparse_r50_fpn_mtst_8x4_1x_qoqo/latest.pth 4 --eval=bbox 1>$expdir/sparse_r50_fpn_mtst_8x4_1x_qoqo/nohup_xxx 2>&1 &

# 查看模型结构
python tools/misc/check_model_arch.py --config=configs/xyz_rcnn/xyz_rcnn_r50_xpn_hlkt.py


# 绘制训练曲线
## 绘制 mAP 曲线
python tools/analysis_tools/analyze_logs.py plot_curve /home/xdata/zhangjp/experiments/mmd-paa-hlkt/r50-fpn-1x-8x4-640px/xxx.log.json --keys bbox_mAP --legend bbox_mAP

## Compare the bbox mAP of two runs in the same figure.
python tools/analysis_tools/analyze_logs.py plot_curve log1.json log2.json --keys bbox_mAP --legend run1 run2
python tools/analysis_tools/analyze_logs.py plot_curve $expdir/x/RUN1.log.json $expdir/xx/RUN2.log.json --keys bbox_mAP --legend RUN1 RUN2 --title Accuracy-Curves-During-Training

## 绘制 loss 曲线
python tools/analysis_tools/analyze_logs.py plot_curve /home/xdata/zhangjp/experiments/mmd-fovea-hlkt/r50-fpn-4x4-1x-800px/20210610_180022.log.json --keys loss_cls loss_bbox loss --legend loss_cls loss_bbox loss

## 计算FLOPs和Parameters

## 绘制混淆矩阵
python tools/analysis_tools/confusion_matrix.py ${CONFIG}  ${DETECTION_RESULTS}  ${SAVE_DIR} --show
https://github.com/open-mmlab/mmdetection/pull/6344
https://github.com/RangiLyu/mmdetection/blob/conf_matrix/tools/analysis_tools/confusion_matrix.py

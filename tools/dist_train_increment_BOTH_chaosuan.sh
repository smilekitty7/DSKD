#!/usr/bin/env bash

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
#export CUDA_VISIBLE_DEVICES=4,5,6,7

startTime=`date +"%Y-%m-%d %H:%M:%S"`


# Ali
# nohup tools/dist_train_increment_BOTH_chaosuan.sh 1>/home/softlink/kmxexp/il_learning/gfl_deformable_detr_40_r50_8x4_1x_qoqo_hard+corr+decode_v22/nohup 2>&1 &
# tail -f /home/softlink/kmxexp/il_learning/gfl_deformable_detr_40_r50_8x4_1x_qoqo_hard+corr+decode_v22/nohup
CONFIG=${1:-'/home/zhangjp/projects/incremental_mmdet/configs/deformable_detr/chaosuan_gfl_deformable_detr_70_r50_8x4_1x_qoqo_il.py'}
WORKDIR=${2:-'/home/softlink/kmxexp/il_learning/gfl_deformable_detr_40_r50_8x4_1x_qoqo_hard+corr+decode_v22/'}

GPUS=${3:-4}
CHECKPOINT=${4:-''}

PORT=${PORT:-32913}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
          $(dirname "$0")/train_increment.py \
          --config=$CONFIG \
          --work-dir=$WORKDIR \
          --resume-from=$CHECKPOINT \
          --launcher=pytorch ${@:3}


endTime=`date +"%Y-%m-%d %H:%M:%S"`
st=`date -d  "$startTime" +%s`
et=`date -d  "$endTime" +%s`

sumHours=$((($et-$st)/3600))
sumMinutes=$((($et-$st)%60))
echo "运行总时间: $sumHours 小时，$sumMinutes 分钟."

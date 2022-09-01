#!/usr/bin/env bash

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
# export CUDA_VISIBLE_DEVICES=0, 1
startTime=`date +"%Y-%m-%d %H:%M:%S"`


CONFIG=${1:-'/home/zhangjp/projects/now-projects/xmdet220/configs/yoloy/yoloy_resnet_qoqo.py'}
WORKDIR=${2:-'/home/softlink/zhjpexp/yoloy-r18-stst-qoqo'}
GPUS=${3:-4}
CHECKPOINT=${4:-'/home/softlink/zhjpexp/yoloy-r18-stst-qoqo/latest.pth'}


PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
          $(dirname "$0")/xxtrain.py \
          --config=$CONFIG \
          --resume-from=$CHECKPOINT \
          --work-dir=$WORKDIR \
          --launcher=pytorch ${@:3}

endTime=`date +"%Y-%m-%d %H:%M:%S"`
st=`date -d  "$startTime" +%s`
et=`date -d  "$endTime" +%s`

sumHours=$((($et-$st)/3600))
sumMinutes=$((($et-$st)%60))
echo "运行总时间: $sumHours 小时，$sumMinutes 分钟."


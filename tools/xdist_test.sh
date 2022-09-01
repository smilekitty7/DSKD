#!/usr/bin/env bash

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
# export CUDA_VISIBLE_DEVICES=0, 1

startTime=`date +"%Y-%m-%d %H:%M:%S"`

#CONFIG=${1:-'/home/zhangjp/projects/now-projects/xmdet220/configs/yolof/yolof_resnet_qoqo_il.py'}
#CHECKPOINT=${2:-'/home/softlink/zhjpexp/yolof-r18-stst-qoqo-il80-v4/epoch_12.pth'}
#CONFIG=${1:-'/home/zhangjp/projects/now-projects/xmdet220/configs/deformable_detr/deformdetr_resnet_qoqo.py'}
#CHECKPOINT=${2:-'/home/softlink/zhjpexp/defdetr_mini/epoch_12_alldata.pth'}
CONFIG=${1:-'/home/zhangjp/projects/now-projects/xmdet220/configs/yolox/yolox_resnet_qoqo_il.py'}
CHECKPOINT=${2:-'/home/softlink/zhjpexp/yolox-r18-stst-qoqo-il20-v0/task1_epoch_12.pth'}
WORKDIR=${3:-'/home/xdata/zhangjp/experiments/common_exp'}
GPUS=${4:-4}

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/xxtest.py \
    --config $CONFIG \
    --checkpoint $CHECKPOINT \
    --work-dir $WORKDIR\
    --launcher pytorch \
    ${@:4}


endTime=`date +"%Y-%m-%d %H:%M:%S"`
st=`date -d  "$startTime" +%s`
et=`date -d  "$endTime" +%s`

sumHours=$((($et-$st)/3600))
sumMinutes=$((($et-$st)%60))
echo "运行总时间: $sumHours 小时，$sumMinutes 分钟."
# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import time
import warnings
# import matplotlib.pyplot as plt
# plt.switch_backend('agg')

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint, wrap_fp16_model)

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset, replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.utils import setup_multi_processes, update_data_root


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test (and eval) a model')

    parser.add_argument('--config',
                        # default='../configs/yolox/yolox_resnet_qoqo.py',
                        # default='../configs/deformable_detr/deformdetr_resnet_qoqo.py',
                        # default='../configs/yolox/yolox_resnet_qoqo.py',
                        # default='../configs/yolof/yolof_resnet_qoqo_il.py',
                        # default='../configs/yolox/yolox_resnet_qoqo_il.py',
                        default='../configs/deformable_detr/gfl_deformable_detr_r50_8x4_1x_qoqo_il_vsmall.py',
                        help='train config file path')
    parser.add_argument('--trainval', default='val', help='train|val|test')
    parser.add_argument('--checkpoint',
                        # default='/home/softlink/zhjpexp/yolox_r50_qoqo3w1k_stst/epoch_12.pth',
                        # default='/home/softlink/zhjpexp/defdetr_mini/epoch_12_alldata.pth',
                        # default='/home/softlink/zhjpexp/yolox-r18-stst-qoqo/latest.pth',
                        # default='/home/softlink/zhjpexp/yolox-r18-stst-qoqo-il20-v0/task1_epoch_12.pth',
                        # default='/home/softlink/zhjpexp/yolox_r18_qoqo_il80_all/epoch_12_teacher.pth',
                        default='/home/softlink/experiments/il_learning/gfl_deformable_detr_20_1/epoch_11_20-1.pth',
                        # default='/home/softlink/experiments/il_learning/gfl_deformable_detr_20_2/epoch_12.pth',
                        # default='/home/softlink/experiments/il_learning/gfl_deformable_detr_20_1/HardMemory_task_2_epoch_12.pth',
                        # default='/home/softlink/experiments/il_learning/gfl_deformable_detr_HardMemory/HardMemory_task_3_epoch_1.pth',
                        # default='/home/softlink/zhjpexp/yolof-r18-stst-qoqo-il80-v4/epoch_12.pth',
                        # default='/home/softlink/zhjpexp/yolof-r50-stst-qoqo-il80/epoch_12.pth',
                        # default='/home/softlink/experiments/wrxt-faster-r18-stst/epoch_12.pth',
                        help='checkpoint file')
    parser.add_argument('--show', default=0, action='store_true', help='show results')
    parser.add_argument('--show-dir',
                        default=None and '/home/softlink/experiments/wrxt-yoloy-r18-stst-datav1/detimgs',
                        help='directory where painted images will be saved, 检测图片存放')
    parser.add_argument('--work-dir',
                        default=None and '/home/softlink/experiments/wrxt-yoloy-r18-stst/detlogs',
                        help='the dir to save logs and models，检测日志存放')
    parser.add_argument('--out', default=None, help='output result file in pickle format')

    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase the inference speed')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=[0],    # , 1, 2, 3
        help='id of gpu to use only applicable to non-distributed testing)')
    parser.add_argument(
        '--format-only',
        action='store_true', default=False,
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        default=['bbox', 'proposal'][0],
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options', default={'classwise': True, },
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)

    # update data root according to MMDET_DATASETS
    update_data_root(cfg)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if 'pretrained' in cfg.model:
        cfg.model.pretrained = None
    elif 'init_cfg' in cfg.model.backbone:
        cfg.model.backbone.init_cfg = None

    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    cfg_data_which = getattr(cfg.data, args.trainval)

    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg_data_which, dict):
        cfg_data_which.test_mode = True
        samples_per_gpu = cfg_data_which.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg_data_which.pipeline = replace_ImageToTensor(
                cfg_data_which.pipeline)
    elif isinstance(cfg_data_which, list):
        for ds_cfg in cfg_data_which:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg_data_which])
        if samples_per_gpu > 1:
            for ds_cfg in cfg_data_which:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    rank, _ = get_dist_info()
    # allows not to create
    if args.work_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        json_file = osp.join(args.work_dir, f'eval_{timestamp}.json')

    # build the dataloader
    # print(f'cfg_data_which ==> {cfg_data_which}')
    dataset = build_dataset(cfg_data_which)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.ALL_CLASSES_IDS = dataset.ALL_CLASSES_IDS
        model.ALL_CLASSES = dataset.ALL_CLASSES
        model.PRED_CLASSES = dataset.PRED_CLASSES
        model.LOAD_CLASSES = dataset.LOAD_CLASSES
        model.START_LABEL = dataset.START_LABEL
        model.cat2label = dataset.cat2label
        model.label2cat = dataset.label2cat
        model.CLASSES = dataset.PRED_CLASSES
    # print(model.CLASSES)

    if not distributed:
        model = MMDataParallel(model, device_ids=cfg.gpu_ids)
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir, args.show_score_thr)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir, args.gpu_collect)
    # outputs [ImgNums, CateNums, ObjectNums, 4+1]

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule', 'dynamic_intervals'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            metric = dataset.evaluate(outputs, **eval_kwargs)
            print(metric)
            metric_dict = dict(config=args.config, metric=metric)
            if args.work_dir is not None and rank == 0:
                mmcv.dump(metric_dict, json_file)


if __name__ == '__main__':
    main()

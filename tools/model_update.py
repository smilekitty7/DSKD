# -*- coding: utf-8 -*-  
'''
@author: zhjp   2022/4/5 下午10:21
@file: update_student.py
'''
import datetime
import time
import torch
from torch.nn.parallel import DistributedDataParallel

from pathlib import Path
import json
import copy


def set_teacher_model(model, args=None):
    if isinstance(model, DistributedDataParallel):
        model_wo_ddp = model.module
    else:
        model_wo_ddp = model
    model = model.eval()
    model_wo_ddp = model_wo_ddp.eval()
    return model, model_wo_ddp


def set_student_model(model, args=None):
    if isinstance(model, DistributedDataParallel):
        model_wo_ddp = model.module
    else:
        model_wo_ddp = model
    # if args.distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    #     model_wo_ddp = model.module
    return model, model_wo_ddp












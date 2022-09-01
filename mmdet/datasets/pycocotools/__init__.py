__author__ = 'tylin'

# abcdefg  added by zhjp for increment learning
# 1. 修改了COCOeval中的第135-139行
# 2. 修改了COCO中的getCatIds()、getImgIds()函数

from .cocoeval import COCOeval
from .coco import COCO, _isArrayLike

__all__ = [
    'COCO', 'COCOeval', '_isArrayLike',
]

#!/usr/bin/env python3
# -*-coding: utf-8 -*-
#
help = '[imgfunc] import部'
#

import os
import sys

[sys.path.append(d) for d in ['./Tools/imgfunc/',
                              '../Tools/imgfunc/'] if os.path.isdir(d)]
from read_write import getCh, read, readN, write, isImgPath
from blank_img import white, black, blank
from convert_img import cleary, encodeDecode, encodeDecodeN, cut, cutN, splitSQ, splitSQN
from convert_img import rotate, rotateR, rotateRN, flip, flipR, flipN
from convert_img import resize, resizeP, resizeN, size2x
from paste import paste
from arr import arr2x, arrNx, img2arr, imgs2arr, arr2img, arr2imgs

#!/usr/bin/env python3
# -*-coding: utf-8 -*-
#
help = '[imgfunc] import部'
#

import os
import sys

[sys.path.append(d) for d in ['./Tools/imgfunc/',
                              '../Tools/imgfunc/'] if os.path.isdir(d)]

from . import read_write as io
from . import blank_img as blank
from . import convert_img as cnv
from . import paste
from . import arr

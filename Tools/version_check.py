#!/usr/bin/env python3
# -*-coding: utf-8 -*-
#
help = '各ライブラリのバージョンを表示する'
#

import os
import sys

[sys.path.append(d) for d in ['./Tools/', '../Tools/'] if os.path.isdir(d)]
from func import getPythonVer


def main():
    py = 'Python'
    cv = 'OpenCV'
    np = 'Numpy'
    ch = 'Chainer'
    cp = 'Cupy'
    idp = 'iDeep'
    mpl = 'Matplotlib'
    pil = 'Pillow'

    ver = {py: None, cv: None, np: None, ch: None,
           cp: None, idp: None, mpl: None, pil: None}

    ver[py] = getPythonVer()

    try:
        import cv2
        ver[cv] = cv2.__version__
    except:
        pass

    try:
        import numpy
        ver[np] = numpy.__version__
    except:
        pass

    try:
        import chainer
        ver[ch] = chainer.__version__
    except:
        pass

    try:
        import cupy
        ver[cp] = cupy.__version__
    except:
        pass

    try:
        import ideep4py
        ver[idp] = '1.0.4'  # ideep4py.__version__
    except:
        pass

    try:
        import matplotlib
        ver[mpl] = matplotlib.__version__
    except:
        pass

    try:
        import PIL
        ver[pil] = PIL.__version__
    except:
        pass

    [print('- **{}** {}'.format(key, val)) for key, val in ver.items()]


if __name__ == '__main__':
    main()

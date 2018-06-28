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

    import pkg_resources as pkg
    check = ('Python', 'pip', 'numpy', 'opencv-python',
             'chainer', 'chainercv', 'cupy', 'ideep4py', 'matplotlib', 'Pillow')
    ver = {dist.project_name: dist.version
           for dist in pkg.working_set if dist.project_name in check}

    [ver.update({i: None}) for i in check if ver.get(i) is None]
    ver['Python'] = getPythonVer()
    [print('- **{}** {}'.format(key, ver[key])) for key in check]


if __name__ == '__main__':
    main()

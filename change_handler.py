#!/usr/bin/env python3
# -*-coding: utf-8 -*-
#
help = 'ファイルの更新を監視する'
#

import os
from watchdog.events import FileSystemEventHandler


class ChangeHandler(FileSystemEventHandler):

    def __init__(self):
        pass

    def on_created(self, event):
        filepath = event.src_path
        filename = os.path.basename(filepath)
        _, ext = os.path.splitext(filename)
        return filepath, filename, ext

    def on_modified(self, event):
        filepath = event.src_path
        filename = os.path.basename(filepath)
        _, ext = os.path.splitext(filename)
        return filepath, filename, ext

    def on_deleted(self, event):
        filepath = event.src_path
        filename = os.path.basename(filepath)
        _, ext = os.path.splitext(filename)
        return filepath, filename, ext

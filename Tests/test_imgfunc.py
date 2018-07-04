#!/usr/bin/env python3
# -*-coding: utf-8 -*-
#
help = 'imgfuncのテスト用コード'
#

import logging
# basicConfig()は、 debug()やinfo()を最初に呼び出す"前"に呼び出すこと
logging.basicConfig(format='%(message)s')
level = logging.INFO
logging.getLogger('Tools').setLevel(level=level)

import cv2
import unittest
import numpy as np

import imgfunc as I

lenna_path = './Tests/Lenna.bmp'
mandrill_path = './Tests/Mandrill.bmp'


class TestImgFunc(unittest.TestCase):

    def test_getCh(self):
        self.assertEqual(I.io.getCh(-1), cv2.IMREAD_UNCHANGED)
        self.assertEqual(I.io.getCh(0), cv2.IMREAD_UNCHANGED)
        self.assertEqual(I.io.getCh(1), cv2.IMREAD_GRAYSCALE)
        self.assertEqual(I.io.getCh(2), cv2.IMREAD_UNCHANGED)
        self.assertEqual(I.io.getCh(3), cv2.IMREAD_COLOR)
        self.assertEqual(I.io.getCh(4), cv2.IMREAD_UNCHANGED)
        self.assertEqual(I.io.getCh(2.5), cv2.IMREAD_UNCHANGED)

    def test_read(self):
        self.assertEqual(I.io.read(lenna_path).shape, (256, 256, 3))
        self.assertEqual(I.io.read(lenna_path, 0).shape, (256, 256, 3))
        self.assertEqual(I.io.read(lenna_path, 1).shape, (256, 256))
        self.assertEqual(I.io.read(lenna_path, 2).shape, (256, 256, 3))
        self.assertEqual(I.io.read(lenna_path, 3).shape, (256, 256, 3))
        self.assertEqual(I.io.read(lenna_path, 9).shape, (256, 256, 3))
        uso = './testetes.jpg'
        self.assertEqual(I.io.read('./testetes.jpg'), None)

        path = [lenna_path, mandrill_path, lenna_path,
                mandrill_path, lenna_path, mandrill_path]
        self.assertEqual(np.array(I.io.readN(path)).shape, (6, 256, 256, 3))
        self.assertEqual(np.array(I.io.readN(path, 1)).shape, (6, 256, 256))
        path = [lenna_path, mandrill_path, lenna_path, uso,
                mandrill_path, lenna_path, mandrill_path, uso]
        self.assertEqual(np.array(I.io.readN(path)).shape, (6, 256, 256, 3))

    def test_blank(self):
        img = I.blank.blank((128, 128, 3), 0)
        self.assertEqual(img.shape, (128, 128, 3))
        self.assertEqual(np.sum(img), 0)
        with self.assertRaises(SystemExit):
            img = I.blank.blank((128, 128, -1), 0)

        img = I.blank.blank((128, 128, 3), -1)
        self.assertEqual(img.shape, (128, 128, 3))
        self.assertEqual(np.sum(img), 0)
        img = I.blank.blank((128, 128, 1), -1)
        self.assertEqual(img.shape, (128, 128))
        self.assertEqual(np.sum(img), 0)
        img = I.blank.blank((128, 128), -1)
        self.assertEqual(img.shape, (128, 128))
        self.assertEqual(np.sum(img), 0)
        img = I.blank.blank((128, 128, 3), (255, 255, 255))
        self.assertEqual(img.shape, (128, 128, 3))
        self.assertEqual(np.sum(img), 255 * 128 * 128 * 3)
        img = I.blank.blank((128, 128), (255, 255, 255))
        self.assertEqual(img.shape, (128, 128, 3))
        self.assertEqual(np.sum(img), 255 * 128 * 128 * 3)

    def test_isImgPath(self):
        self.assertTrue(I.io.isImgPath(lenna_path))
        self.assertFalse(I.io.isImgPath('./Tools/Tests/Lenno.bmp'))
        self.assertFalse(I.io.isImgPath(None))
        self.assertFalse(I.io.isImgPath(0))

    def test_encodeDecode(self):
        l = cv2.imread(lenna_path)
        m = cv2.imread(mandrill_path)
        self.assertEqual(len(I.cnv.encodeDecodeN([l, m], 3)), 2)
        self.assertEqual(len(I.cnv.encodeDecodeN([l, m], 1)), 2)
        l = cv2.imread(lenna_path, I.cnv.getCh(1))
        m = cv2.imread(mandrill_path, I.cnv.getCh(1))
        self.assertEqual(len(I.cnv.encodeDecodeN([l, m], 3)), 2)
        self.assertEqual(len(I.cnv.encodeDecodeN([l, m], 1)), 2)

    def test_cut(self):
        l = cv2.imread(lenna_path)
        m = cv2.imread(mandrill_path)
        self.assertEqual(I.cnv.cutN([l, m], 64).shape, (2, 64, 64, 3))
        lm16 = [l, l, l, l, m, m, m, m, l, l, l, l, m, m, m, m]
        self.assertEqual(I.cnv.cutN(lm16, 64, 100).shape, (16, 64, 64, 3))
        self.assertEqual(I.cnv.cutN(lm16, 64, 10).shape, (10, 64, 64, 3))
        self.assertEqual(I.cnv.cutN([l, m], 1).shape, (2, 256, 256, 3))
        self.assertEqual(I.cnv.cutN([l, m], 0).shape, (2, 256, 256, 3))
        self.assertEqual(I.cnv.cutN([l, m], -1).shape, (2, 256, 256, 3))

    def test_split(self):
        l = cv2.imread(lenna_path)
        m = cv2.imread(mandrill_path)
        imgs, split = I.cnv.splitSQN([l, m], 32)
        self.assertEqual(imgs.shape, (128, 32, 32, 3))
        self.assertEqual(split, (8, 8))

        imgs, split = I.cnv.splitSQN([l, m], 0)
        self.assertEqual(imgs.shape, (512, 2, 2))
        self.assertEqual(split, (1, 1))

        imgs, split = I.cnv.splitSQN([l, m], 32, 10)
        self.assertEqual(imgs.shape, (120, 32, 32, 3))
        self.assertEqual(split, (8, 8))

        imgs, split = I.cnv.splitSQN([l, m], 32, 100)
        self.assertEqual(imgs.shape, (100, 32, 32, 3))
        self.assertEqual(split, (8, 8))

        imgs, split = I.cnv.splitSQN([l, m], 32, 1000)
        self.assertEqual(imgs.shape, (128, 32, 32, 3))
        self.assertEqual(split, (8, 8))

        print(l.shape, m.shape)
        imgs, split = I.cnv.splitSQN([l, m], 1024)
        self.assertEqual(imgs.shape, (2, 256, 256, 3))
        self.assertEqual(split, (1, 1))

        bk = I.blank.blank((100, 120, 3), 255)
        imgs, split = I.cnv.splitSQN([bk], 1024)
        self.assertEqual(imgs.shape, (1, 100, 100, 3))
        self.assertEqual(split, (1, 1))

        l = cv2.imread(lenna_path, I.io.getCh(1))
        m = cv2.imread(mandrill_path, I.io.getCh(1))
        imgs, split = I.cnv.splitSQN([l, m], 32)
        self.assertEqual(imgs.shape, (128, 32, 32))
        self.assertEqual(split, (8, 8))

    def test_vhstack(self):
        l = cv2.imread(lenna_path)
        m = cv2.imread(mandrill_path)
        self.assertEqual(I.cnv.vhstack([l, m]).shape, (256, 512, 3))
        self.assertEqual(
            I.cnv.vhstack([l, m, l, m], (1, 4)).shape,
            (256, 1024, 3)
        )
        self.assertEqual(I.cnv.vhstack([l, m, l, m]).shape, (256, 1024, 3))
        self.assertEqual(
            I.cnv.vhstack([l, m, l, m], (1, -1)).shape,
            (256, 1024, 3)
        )
        self.assertEqual(
            I.cnv.vhstack([l, m, l, m], (2, -1)).shape,
            (512, 512, 3)
        )
        self.assertEqual(
            I.cnv.vhstack([l, m, l, m, l], (2, -1)).shape,
            (512, 768, 3)
        )
        self.assertEqual(
            I.cnv.vhstack([l, m, l, m, l], (-1, 3)).shape,
            (512, 768, 3)
        )
        self.assertEqual(
            I.cnv.vhstack([l, m, l, m], (1, 1)).shape,
            (256, 1024, 3)
        )

    def test_rotate(self):
        l = cv2.imread(lenna_path)
        m = cv2.imread(mandrill_path)
        imgs, angle = I.cnv.rotateRN([l, m], 3)
        self.assertEqual(imgs.shape, (6, 256, 256, 3))
        self.assertEqual(angle.shape, (6,))

    def test_flip(self):
        l = cv2.imread(lenna_path)
        m = cv2.imread(mandrill_path)
        self.assertEqual(I.cnv.flipN([l, m]).shape, (6, 256, 256, 3))
        self.assertEqual(I.cnv.flipN([l, m], -1).shape, (2, 256, 256, 3))
        self.assertEqual(I.cnv.flipN([l, m], 0).shape,  (2, 256, 256, 3))
        self.assertEqual(I.cnv.flipN([l, m], 1).shape,  (4, 256, 256, 3))
        self.assertEqual(I.cnv.flipN([l, m], 2).shape,  (6, 256, 256, 3))
        self.assertEqual(I.cnv.flipN([l, m], 3).shape,  (8, 256, 256, 3))

    def test_resize(self):
        l = cv2.imread(lenna_path)
        m = cv2.imread(mandrill_path)
        imgs = I.cnv.size2x([l, m])
        self.assertEqual(imgs[0].shape, (512, 512, 3))
        self.assertEqual(imgs[1].shape, (512, 512, 3))
        self.assertEqual(I.cnv.resize(l, -1).shape, (256, 256, 3))
        self.assertEqual(I.cnv.resize(l, 2).shape, (512, 512, 3))
        self.assertEqual(I.cnv.resize(l, 1.5).shape, (384, 384, 3))
        self.assertEqual(I.cnv.resize(l, 0.5).shape, (128, 128, 3))

    def test_imgs2arr(self):
        l = cv2.imread(lenna_path)
        m = cv2.imread(mandrill_path)
        self.assertEqual(I.arr.imgs2arr([l, m]).shape, (2, 3, 256, 256))

        l = cv2.imread(lenna_path, I.io.getCh(1))
        m = cv2.imread(mandrill_path, I.io.getCh(1))
        self.assertEqual(I.arr.imgs2arr([l, m]).shape, (2, 1, 256, 256))


if __name__ == '__main__':
    unittest.main()

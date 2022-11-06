#!/usr/bin/env python3
"""
Copyright 2020, Zixin Luo, HKUST.
Image matching example.
"""
import yaml
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from utils.opencvhelper import MatcherWrapper

from models import get_model


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('config', None, """Path to the configuration file.""")


def load_imgs(img_paths, max_dim):
    rgb_list = []
    gray_list = []
    for img_path in img_paths:
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[..., np.newaxis]
        img = img[..., ::-1]
        rgb_list.append(img)
        gray_list.append(gray)
    return rgb_list, gray_list


def extract_local_features(gray_list, model_path, config):
    model = get_model('feat_model')(model_path, **config)
    descs = []
    kpts = []
    for gray_img in gray_list:
        desc, kpt, _ = model.run_test_data(gray_img)
        print('feature_num', kpt.shape[0])
        descs.append(desc)
        kpts.append(kpt)
    return descs, kpts


def main(argv=None):  # pylint: disable=unused-argument
    """Program entrance."""
    # parse input
    with open(FLAGS.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    # load testing images.
    img_path=[]
    #I modified this 069
    for i in range(2,15):
      img_path.append("dataset/{}-{}.jpg".format(i,1))
      img_path.append("dataset/{}-{}.jpg".format(i,2))
    #I modified this 069
    
    #rgb_list, gray_list = load_imgs(config['img_paths'], config['net']['max_dim'])
    rgb_list, gray_list = load_imgs(img_path, config['net']['max_dim'])
    # extract regional features.
    descs, kpts = extract_local_features(gray_list, config['model_path'], config['net'])
    # feature matching and draw matches.
    matcher = MatcherWrapper()
    #I modified this 205121069
    i=0
    j=1
    while j<len(rgb_list):
      match, mask = matcher.get_matches(
        descs[i], descs[j], kpts[i], kpts[j],
        ratio=config['match']['ratio_test'], cross_check=config['match']['cross_check'],
        err_thld=3, ransac=True, info='ASLFeat')
      # draw matches
      disp = matcher.draw_matches(rgb_list[i], kpts[i], rgb_list[j], kpts[j], match, mask)

      output_name = 'disp{}.jpg'.format(int(j/2)+1)
      print('image save to', output_name)
      plt.imsave(output_name, disp)
      i += 2
      j += 2
      #I modified this 069
if __name__ == '__main__':
    tf.compat.v1.app.run()

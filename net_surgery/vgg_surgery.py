#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os, sys, optparse

from deepseg import GlobalVariable as gv
import caffe

optparser = optparse.OptionParser()
optparser.add_option("-i", "--input", dest="input", default="vgg16.caffemodel", help="Input vgg 16 network parameters filepath")
optparser.add_option("-o", "--output", dest="output", default="vgg16.fc.caffemodel", help="Output vgg 16 network parameters filepath")
(opts, _)= optparser.parse_args()

script_path = os.path.realpath(__file__)
script_name = script_path.split('/')[-1]
cur_dirpath = script_path[:-len(script_name)]
print "Curent Directory Path: {}".format(cur_dirpath)

proto_vgg16   =os.path.join( cur_dirpath, "vgg16.prototxt")
proto_vgg16_fc=os.path.join( cur_dirpath, "vgg16_fc.prototxt")

def main(in_filename, out_filename):
  in_model_filepath  = os.path.join( gv.model_path, in_filename  )
  out_model_filepath = os.path.join( gv.model_path, out_filename )

  net_vgg16 = caffe.Net(proto_vgg16, in_model_filepath, caffe.TEST)
  params = ['fc6', 'fc7', 'fc8']

  fc_params = { pr: (net_vgg16.params[pr][0].data, net_vgg16.params[pr][1].data) for pr in params }

  for fc in params:
    print '{} weights are {} dimensional and biases are {} dimensional'.format(fc, fc_params[fc][0].shape, fc_params[fc][1].shape)

  net_vgg16_fc = caffe.Net(proto_vgg16_fc, in_model_filepath, caffe.TEST)

  params_full_conv = ['fc6-conv', 'fc7-conv', 'fc8-conv']
  conv_params = {pr: (net_vgg16_fc.params[pr][0].data, net_vgg16_fc.params[pr][1].data) for pr in params_full_conv}

  for conv in params_full_conv:
    print '{} weights are {} dimensional and biases are {} dimensional'.format(conv, conv_params[conv][0].shape, conv_params[conv][1].shape)

  for pr, pr_conv in zip(params, params_full_conv):
    conv_params[pr_conv][0].flat = fc_params[pr][0].flat  # flat unrolls the arrays
    conv_params[pr_conv][1][...] = fc_params[pr][1]

  print "Writing out model to {}...".format(out_model_filepath)
  net_vgg16_fc.save(out_model_filepath)


if __name__ == "__main__":
  main(opts.input, opts.output)

  print "Done."
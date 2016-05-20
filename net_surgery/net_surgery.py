# Make sure that caffe is on the python path:
#caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
import os, sys, optparse
import sys
#sys.path.insert(0, caffe_root + 'python')

import caffe


optparser = optparse.OptionParser()
optparser.add_option("-f", "--fold", dest="fold", default="1", help="Base Dir")
(opts, _)= optparser.parse_args()

def net_surgery(fold):
  # Load the original network and extract the fully connected layers' parameters.
  net = caffe.Net('action/fold_' + fold + '_action_train_val_finetune.prototxt', 
                'action/fold_' + fold + '_action_snapshots_iter_50000.caffemodel', 
                caffe.TRAIN)

  # Initialize weights and biases
  params_keys = net.params.keys()
  print params_keys
  n_params = {pr: (net.params[pr][0].data, net.params[pr][1].data) for pr in params_keys}  # n_params = {name: (weights, biases)}
  blobs_keys = net.blobs.keys()
  n_blobs = {pr: (net.blobs[pr].data) for pr in blobs_keys}
  #print n_blobs


  # Load the original network and extract the fully connected layers' parameters.
  frame_net = caffe.Net('frame/fold_' + fold + '_frame_train_val_finetune.prototxt', 
                'frame/fold_' + fold + '_frame_snapshots_iter_50000.caffemodel', 
                caffe.TRAIN)

  # Initialize weights and biases
  frame_params_keys = frame_net.params.keys()
  print frame_params_keys
  frame_n_params = {pr: (frame_net.params[pr][0].data, frame_net.params[pr][1].data) for pr in frame_params_keys}  # n_params = {name: (weights, biases)}
  frame_blobs_keys = frame_net.blobs.keys()
  frame_n_blobs = {pr: (frame_net.blobs[pr].data) for pr in frame_blobs_keys}


  # Load the original network and extract the fully connected layers' parameters.
  pose_net = caffe.Net('pose/fold_' + fold + '_pose_train_val_finetune.prototxt', 
                'pose/fold_' + fold + '_pose_snapshots_iter_50000.caffemodel',
                caffe.TRAIN)

  # Initialize weights and biases
  pose_params_keys = pose_net.params.keys()
  print pose_params_keys
  pose_n_params = {pr: (pose_net.params[pr][0].data, pose_net.params[pr][1].data) for pr in pose_params_keys}  # n_params = {name: (weights, biases)}
  pose_blobs_keys = pose_net.blobs.keys()
  pose_n_blobs = {pr: (pose_net.blobs[pr].data) for pr in pose_blobs_keys}




  ns_net = caffe.Net('fold_' + fold + '/train_val_fold_' + fold + '.prototxt',caffe.TRAIN)
  ns_params_keys = ns_net.params.keys()
  print ns_params_keys
  ns_params = {pr: (ns_net.params[pr][0].data, ns_net.params[pr][1].data) for pr in ns_params_keys}
  # ns_params = {pr: (ns_net.params[pr][0].data) for pr in ns_params_keys}
  length = len(params_keys)
  ns_length = len(ns_params_keys)
  print length,ns_length
  for i in range(ns_length):
	j = i % length
	k = (i - j) / length
	ns_pr = ns_params_keys[i]
	if k == 0: 
		pr = params_keys[j]
		print j,ns_pr,pr,ns_params[ns_pr][0].shape,n_params[pr][0].shape
		ns_params[ns_pr][0].flat = n_params[pr][0].flat
		ns_params[ns_pr][1].flat = n_params[pr][1].flat
	elif k == 2:
		pr = frame_params_keys[j]
		print j,ns_pr,"frame_"+pr,ns_params[ns_pr][0].shape,frame_n_params[pr][0].shape
		ns_params[ns_pr][0].flat = frame_n_params[pr][0].flat
		ns_params[ns_pr][1].flat = frame_n_params[pr][1].flat
	else:
		pr = pose_params_keys[j]
		print j,ns_pr,"pose_"+pr,ns_params[ns_pr][0].shape,pose_n_params[pr][0].shape
		ns_params[ns_pr][0].flat = pose_n_params[pr][0].flat
		ns_params[ns_pr][1].flat = pose_n_params[pr][1].flat


  # Initialize filters
  ns_blobs_keys = ns_net.blobs.keys()
  ns_blobs = {pr: (ns_net.blobs[pr].data) for pr in ns_blobs_keys}
  length = len(blobs_keys)
  ns_length = len(ns_blobs_keys)
  for i in range(ns_length):
	j = i % length
	k = (i - j) / length
	ns_pr = ns_blobs_keys[i]
	if k == 0:
		pr = blobs_keys[j]
		#print j,ns_pr,pr,ns_blobs[ns_pr].shape,n_blobs[pr].shape
		ns_blobs[ns_pr].flat = n_blobs[pr].flat
		#print j,ns_pr,pr,ns_blobs[ns_pr].shape,n_blobs[pr].shape
	elif k == 2:
		pr = frame_blobs_keys[j]
		#print j,ns_pr,pr,ns_blobs[ns_pr].shape,n_blobs[pr].shape
		ns_blobs[ns_pr].flat = frame_n_blobs[pr].flat
		#print j,ns_pr,pr,ns_blobs[ns_pr].shape,n_blobs[pr].shape
	else:
		pr = pose_blobs_keys[j]
		#print j,ns_pr,pr,ns_blobs[ns_pr].shape,n_blobs[pr].shape
		ns_blobs[ns_pr].flat = pose_n_blobs[pr].flat
		#print j,ns_pr,pr,ns_blobs[ns_pr].shape,n_blobs[pr].shape


  ns_net.save('fold_' + fold + '/net_surgery_fold_' + fold + '.caffemodel')

if __name__=="__main__":
  net_surgery(opts.fold)


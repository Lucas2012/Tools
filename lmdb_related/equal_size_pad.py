import numpy as np 
import caffe
from caffe.proto import caffe_pb2
import scipy.io
import os, sys, optparse
import lmdb
import h5py


optparser = optparse.OptionParser()
optparser.add_option("-i", "--databasein", dest="databasein", default="/cs/vml2/shared/data/deepseg/database/pascal/lmdb/pascal.train.lmdb", help="Base Dir")
optparser.add_option("-o", "--databaseout", dest="databaseout", default="/cs/vml2/shared/data/deepseg/database/pascal/lmdb/pascal_paded.train.lmdb", help="Base Dir")
optparser.add_option("-r", "--databaseratio", dest="databaseratio", default="/cs/vml2/shared/data/deepseg/database/pascal/hdf5/pascal_ratio.train.hdf5", help="Base Dir")
optparser.add_option("-g", "--databasegtin", dest="databasegtin", default="/cs/vml2/shared/data/deepseg/database/pascal/lmdb/pascal.context.train.lmdb", help="Base Dir")
optparser.add_option("-j", "--databasegtout", dest="databasegtout", default="/cs/vml2/shared/data/deepseg/database/pascal/lmdb/pascal_paded.context.train.lmdb", help="Base Dir")
optparser.add_option("-c", "--databasecoords", dest="databasecoords", default="/cs/vml2/shared/data/deepseg/database/pascal/hdf5/pascal_coords.train.hdf5", help="Base Dir")


(opts, _) = optparser.parse_args()
im_size_h = 500
im_size_w = 500
flip = True

def equal_size_padding(database_path_in, dataset_path_out, dataset_path_ratio, dataset_path_coords, gt_path_in, gt_path_out):
	db_file_in      = lmdb.open(database_path_in, readonly=True)
	gt_file_in      = lmdb.open(gt_path_in, readonly=True)
	db_file_out     = lmdb.open(dataset_path_out, map_size = 60485760000)
	db_file_ratio   = lmdb.open(dataset_path_ratio, map_size = 104857600)
	gt_file_out     = lmdb.open(gt_path_out, map_size = 6048576000)
	db_cursor       = db_file_in.begin().cursor()
	gt_cursor       = gt_file_in.begin().cursor()
	idx = 0
	count = 0
	ratio_all = np.array([])
	coords = np.array([])
	for key, value in db_cursor:
		datum_in = caffe_pb2.Datum()
		datum_in.ParseFromString(value)
		data = np.array( caffe.io.datum_to_array(datum_in) )
		if count % 100 == 0:
			print 'data shape: ', data.shape
		
		Num = 1
		if flip:
			Num = 2
		
		zero_data = np.ones([Num, 3, im_size_h, im_size_w], dtype = np.uint8)
		zero_data[ :, 0, :, : ] *= 104
		zero_data[ :, 1, :, : ] *= 117
		zero_data[ :, 2, :, : ] *= 123
		im_h = data.shape[-2]
		im_w = data.shape[-1]
		zero_data[0, :, 0 : im_h, 0 : im_w] = data
		coords = np.append( coords, [im_h, im_w], axis = 0 )
		if flip:
			zero_data[1, 0, 0 : im_h, im_w- 1::-1] = data[0,:,:]
			zero_data[1, 1, 0 : im_h, im_w- 1::-1] = data[1,:,:]
			zero_data[1, 2, 0 : im_h, im_w- 1::-1] = data[2,:,:]
			coords = np.append( coords, [im_h, im_w], axis = 0 )

		if flip:
			ratio = np.array([np.double (im_h * im_w) / np.double( zero_data.shape[2] * zero_data.shape[3] ),
				np.double (im_h * im_w) / np.double( zero_data.shape[2] * zero_data.shape[3] )])
		else:
			ratio = np.array(np.double (im_h * im_w) / np.double( zero_data.shape[2] * zero_data.shape[3] ))
		ratio_all = np.append(ratio_all, ratio)

		for n in range(zero_data.shape[0]):
			datum = caffe.proto.caffe_pb2.Datum()
			datum.channels = zero_data.shape[1]
			datum.height   = zero_data.shape[2]
			datum.width    = zero_data.shape[3]
			datum.data     = zero_data[n,:].tobytes()
			datum.label    = int(datum_in.label)

			str_id         = '{:08}'.format(idx)

			with db_file_out.begin(write=True) as txn2:
				# txn is a Transaction object
				# The encode is only essential in Python 3
				txn2.put(str_id.encode('ascii'), datum.SerializeToString())

			
			'''datum_ratio = caffe.proto.caffe_pb2.Datum()
			datum_ratio.channels = 1
			datum_ratio.height   = 1
			datum_ratio.width    = 1
			datum_ratio.data     = ratio[n].tobytes()
			datum_ratio.label    = int(datum_in.label)

			str_id         = '{:08}'.format(idx)

			with db_file_ratio.begin(write=True) as txn3:
				# txn is a Transaction object
				# The encode is only essential in Python 3
				txn3.put(str_id.encode('ascii'), datum_ratio.SerializeToString())'''
			idx += 1
		count += 1 


	coords = np.reshape(coords, [len(coords) / 2, 2])
	coords_h5db = dataset_path_coords
	ratio_h5db  = dataset_path_ratio

	if not os.path.exists(coords_h5db):
		os.makedirs(coords_h5db)
	print os.path.exists(coords_h5db), coords_h5db
	with h5py.File(os.path.join(coords_h5db, 'train.h5'), 'w') as hf:
		hf.create_dataset('coords',data = np.array(coords))
	with open(os.path.join(coords_h5db, "train.txt"), 'w') as f:
		f.write( os.path.join(coords_h5db, 'train.h5') + '\n' )

	if not os.path.exists(ratio_h5db):
		os.makedirs(ratio_h5db)
	with h5py.File(os.path.join(ratio_h5db, 'train.h5'), 'w') as hf:
		hf.create_dataset('ratio',data = np.array(ratio_all))
	with open(os.path.join(ratio_h5db, "train.txt"), 'w') as f:
		f.write( os.path.join(ratio_h5db, 'train.h5') + '\n' )

	idx = 0
	for key_gt, value_gt in gt_cursor:
		Num = 1
		if flip:
			Num = 2
		gt_in = caffe_pb2.Datum()
		gt_in.ParseFromString(value_gt)
		gt_data = np.array( caffe.io.datum_to_array(gt_in) )
		im_h = gt_data.shape[-2]
		im_w = gt_data.shape[-1]
		print 'origin unique', np.unique( gt_data ).astype(np.float32) 

		if count % 100 == 0:
			print 'gt data shape: ', gt_data.shape

		gt_dummy = np.ones([Num, 1, im_size_h, im_size_w], dtype = np.uint8) * 255
		gt_dummy[0, :, 0 : im_h, 0 : im_w] = gt_data

		print 'paded unique', np.unique( gt_dummy ).astype(np.int32) 

		if flip:
			gt_dummy[1, 0, 0 : im_h, im_w - 1::-1] = gt_data[0,:,:]

		for n in range(gt_dummy.shape[0]):
			datum_gt = caffe.proto.caffe_pb2.Datum()
			datum_gt.channels = gt_dummy.shape[1]
			datum_gt.height   = gt_dummy.shape[2]
			datum_gt.width    = gt_dummy.shape[3]
			datum_gt.data     = gt_dummy[n].tobytes()
			datum_gt.label    = int(gt_in.label)

			str_id         = '{:08}'.format(idx)

			with gt_file_out.begin(write=True) as txn4:
				# txn is a Transaction object
				# The encode is only essential in Python 3
				txn4.put(str_id.encode('ascii'), datum_gt.SerializeToString())

			idx += 1
		count += 1


if __name__ == "__main__":
	equal_size_padding(opts.databasein, opts.databaseout, opts.databaseratio, opts.databasecoords, opts.databasegtin, opts.databasegtout)

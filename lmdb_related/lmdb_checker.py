import os, sys, optparse
import scipy.io
import lmdb
import numpy as np
from caffe.proto import caffe_pb2
import caffe

optparser = optparse.OptionParser()
optparser.add_option("-d", "--database", dest="database", default="", help="Base Directory")
(opts, _)= optparser.parse_args()

def lmdb_checker(lmdb_path):
	db_file   = lmdb.open(lmdb_path, readonly=True)
	db_cursor = db_file.begin().cursor()

	datum = caffe_pb2.Datum()

	for key, value in db_cursor:
		datum.ParseFromString(value)
		data = np.array( caffe.io.datum_to_array(datum) )
		print 'First data shape: ', data.shape
		break

if __name__ == "__main__":
	lmdb_checker(opts.database)


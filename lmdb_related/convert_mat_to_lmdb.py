import os, sys, optparse
import scipy.io
import lmdb
import numpy as np
from caffe.proto import caffe_pb2
import caffe


optparser = optparse.OptionParser()
optparser.add_option("-b", "--base", dest="base", default="/cs/vml2/gza11/data/dmmc/", help="Base Dir")
optparser.add_option("-m", "--mat", dest="mat", default="kmeans_init.mat", help="Output data name")
optparser.add_option("-d", "--dataset", dest="dataset", default="awa", help="Input dataset name")

(opts, _)= optparser.parse_args()

def convert_mat_to_lmdb(basepath, dataset, matfile):
  data_path = os.path.join(basepath, 'results', dataset, 'init')

  mat_path  = os.path.join(data_path, matfile)
  lmdb_path = os.path.join(data_path, 'database_{0}_lmdb'.format(matfile.split('.mat')[0]) )

  mat = scipy.io.loadmat(mat_path)
  results = mat['results']
  labels  = results['labels'][0, 0]

  (N, _) = labels.shape
  print "Start creating initialization database for kmeans...{0} instances.".format( N )

  dummy_data = np.zeros((N, 3, 2, 2))

  db_file   = lmdb.open(lmdb_path)

  sys.stdout.write("Start writing database to {}.".format(lmdb_path))
  for idx in xrange(N):
    if idx % 100 == 0:
      sys.stdout.write('.')
      sys.stdout.flush()

    datum = caffe.proto.caffe_pb2.Datum()
    datum.channels = dummy_data.shape[1]
    datum.height   = dummy_data.shape[2]
    datum.width    = dummy_data.shape[3]
    datum.data     = dummy_data[idx].tobytes()
    datum.label    = int(labels[idx])

    str_id         = '{:08}'.format(idx)

    with db_file.begin(write=True) as txn:
        # txn is a Transaction object
        # The encode is only essential in Python 3
        txn.put(str_id.encode('ascii'), datum.SerializeToString())

  sys.stdout.write("Done.\n")

if __name__=="__main__":
  convert_mat_to_lmdb(opts.base, opts.dataset, opts.mat)

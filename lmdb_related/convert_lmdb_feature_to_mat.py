import os, sys, optparse
import scipy.io
import lmdb
import numpy as np
from caffe.proto import caffe_pb2
import caffe

optparser = optparse.OptionParser()
optparser.add_option("-b", "--base", dest="base", default="/cs/vml2/gza11/data/dmmc/data", help="Base Directory")
optparser.add_option("-o", "--output", dest="output", default="data.random.mat", help="Output data name")
optparser.add_option("-d", "--dataset", dest="dataset", default="indoor", help="Input dataset name")
optparser.add_option("-i", "--input", dest="input", default="indoor.random", help="Input data name")

optparser.add_option("-n", "--num", dest="num", type=int, default=int(6700), help="Number of instances")
optparser.add_option("-f", "--feat", dest="feat", type=int, default=int(4096), help="Dimensionality of feature")
(opts, _)= optparser.parse_args()

def convert_feature(base_path, dataset, in_filename, out_filename, N, F):
  feature_path = os.path.join( base_path, 'feature', dataset, "{}_lmdb".format(in_filename) )

  assert os.path.isdir(base_path),    "Base path not exists."
  assert os.path.isdir(feature_path), "Feature path not exists."

  data  = np.zeros((N, F), dtype=np.float32 )
  label = np.zeros( N, dtype=np.int64 )

  sys.stdout.write('Open Database..{0}.\n'.format(feature_path))

  db_file   = lmdb.open(feature_path, readonly=True)
  db_cursor = db_file.begin().cursor()

  sys.stdout.write('Done.\n')


  datum = caffe_pb2.Datum()
  idx = 0
  sys.stdout.write('Start extracting feature')
  for key, value in db_cursor:
    datum.ParseFromString(value)
    data[idx, :]  = np.squeeze( caffe.io.datum_to_array(datum) )

    idx += 1

    if idx % 1000 == 0:
      sys.stdout.write( '.' )
      sys.stdout.flush()

  sys.stdout.write('\nDone.\n')

  label_filename = "{0}.label".format(in_filename)
  label_path = os.path.join( base_path, 'dataset', dataset, label_filename )
  sys.stdout.write('Load label: {0}.\n'.format(label_path))
  with open(label_path, 'r') as label_file:
    lines  = label_file.readlines()
    for idx, line in enumerate(lines):
      (_, lname) = line.strip().split(' ')
      label[idx] = int(lname)

  sys.stdout.write('Done.\n')

  out_mat = dict()
  out_mat['data']  = data
  out_mat['label'] = label

  out_path = os.path.join( base_path, 'feature', dataset, out_filename )

  sys.stdout.write( 'Dump mat file: {0}.\n'.format(out_path) )
  scipy.io.savemat( out_path, out_mat )
  sys.stdout.write( 'Done.\n' )




if __name__ == '__main__':
    convert_feature(opts.base, opts.dataset,  opts.input, opts.output, int(opts.num), int(opts.feat) )

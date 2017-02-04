import caffe
from caffe.proto import caffe_pb2
import sys, os, optparse
import lmdb
import h5py
import numpy as np

optparser = optparse.OptionParser()
optparser.add_option("-i", "--input", dest="model", default='', help="input lmdb")
optparser.add_option("-o", "--output", dest="output", default='', help="output hdf5")
optparser.add_option("-k", "--key", dest="key", default='fc7', help="kdf5 key")

(opts, _)= optparser.parse_args()

def convert_feature(lmdbdir, hdf5dir, key):
    print 'start transform data...'

    os.system('rm -r %s'%(hdf5dir))
    os.system('mkdir %s'%(hdf5dir))

    lmdb_env = lmdb.open(lmdbdir)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    datum = caffe.proto.caffe_pb2.Datum()

    fc7 = []
    for key, value in lmdb_cursor:
      datum.ParseFromString(value)
      data = caffe.io.datum_to_array(datum)
      if len(fc7) == 0:
        fc7 = np.array(data)
      else
        fc7 = np.append(fc7, np.array(data))
        print fc7.shape
        break

    train_filename = os.path.join(hdf5dir)
    os.system('rm %s'%train_filename)
    with h5py.File(train_filename, 'w') as hf:
      hf.create_dataset(key, data = fc7)
    f.write(train_filename + '\n')

    print '\nDone!'

if __name__=="__main__":
  convert_feature(opts.input, opts.output, opts.key)

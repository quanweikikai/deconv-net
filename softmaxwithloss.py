import caffe
import numpy as np
import os


class softmaxwithloss(caffe.Layer):
    """
    Compute the Euclidean Loss in the same manner as the C++ EuclideanLossLayer
    to demonstrate the class interface for developing layers in Python.
    """

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs.")
        #if len(top) != 2:
        #    raise Exception("Need two outputs.")

    def reshape(self, bottom, top):
        # difference is shape of inputs
        #self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)
        #top[1].data[...] = np.zeros_like(bottom[0].data, dtype=np.float32)
        #self.diff[...] = 0.

    def forward(self, bottom, top):
        self.expArr = bottom[0].data[...]
        rowMax = np.max(self.expArr,axis=1)
        for ll in xrange(self.expArr.shape[0]):
            self.expArr[ll] = self.expArr[ll] - rowMax[ll]
        self.expArr = np.exp(self.expArr)
        sumArr = np.sum(self.expArr,axis=1)
        for ii in xrange(sumArr.shape[0]):
            self.expArr[ii] = self.expArr[ii]/sumArr[ii]
        loss = 0
        for jj in xrange(bottom[1].data[...].shape[0]):
            loss -= np.log(self.expArr[jj,int(bottom[1].data[jj])])
        loss /= bottom[0].data.shape[0]
        self.diff = np.array([loss])
        top[0].data[...] = loss
        #top[1].data[...] = self.expArr

    def backward(self, top, propagate_down, bottom):
        bottom[0].diff[...] = self.expArr[...]
        for ii in xrange(bottom[0].data[...].shape[0]):
            for kk in xrange(1):
                if not propagate_down[kk]:
                    continue
                for jj in xrange(bottom[kk].data[...].shape[1]):
                    if jj == int(bottom[1].data[ii]):
                        bottom[kk].diff[ii,jj] -=1
        if not os.path.isfile('exp.npy'):
            np.save("exp.npy",bottom[1].data)
            np.save("bottom0.npy",bottom[0].diff)
           


        


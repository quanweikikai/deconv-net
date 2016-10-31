import caffe
import matplotlib.pyplot as plt
import numpy as np
import sys

def plotAll(inputArr, plotShape):
    for ii in xrange(plotShape[0]):
        startImg = inputArr[0,ii*plotShape[1]+1,...]
        for jj in xrange(1,plotShape[1]):
            startImg = np.append(startImg,inputArr[0,ii*plotShape[1]+jj,...],axis=0)
        if (ii == 0):
            lineImg = startImg
        else:
            lineImg = np.append(lineImg,startImg,axis=1)
    return lineImg

net = caffe.Net('../deconvTest/lenet.prototxt','../caffe/examples/mnist/lenet_iter_10000.caffemodel',caffe.TEST)
invNet = caffe.Net('../deconvTest/inverseLenet.prototxt',caffe.TEST)
for layer in invNet.params:
    invNet.params[layer][0].data[...] = net.params[layer[2:]][0].data
f, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)

c=np.array( [0.299, 0.58, 0.114] )
a = net.params['conv1'][0].data
img = caffe.io.load_image(sys.argv[1])
img = np.dot(img,c)
img = caffe.io.resize(img,(28,28)) * 255
#forward
net.blobs['test1'].data[0] = img
output = net.forward(['conv1','conv2','SoftmaxOut','ip2'])
#inverse forward
invNet.blobs['input1'].data[0] = output['conv2']
inverseOutput = invNet.forward(['deconv2'])

input2 = inverseOutput['deconv2'][0,...]
input2 = input2/float(np.max(input2))

input2 = caffe.io.resize(input2,(20,24,24))
tmp = np.zeros((1,20,24,24))
tmp[0] = input2

invNet.blobs['input2'].data[0] = tmp
invResult = invNet.forward(['result'])

plotImg1 = plotAll(output['conv2'],(10,5))
plotImg2 = plotAll(output['conv1'],(5,4))
plotImg3 = invResult['result']

ax1.imshow(plotImg1)
ax2.imshow(plotImg2)
ax4.imshow(plotImg3[0,0,...]*255)
ax3.imshow(img)
print output['SoftmaxOut']
print output['ip2']
plt.show()
	

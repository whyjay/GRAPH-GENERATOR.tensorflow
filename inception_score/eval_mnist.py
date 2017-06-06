import model_mnist as model
import numpy as np
import os
from IPython import embed

exps = [
    'ICCV2017/samples/EXP38/dcgan/mnist/noise/1490076837',
]
maxs = []

for exp in exps:
    #base_dir = '/data/whyjay/NIPS2017/'
    base_dir = '/data/whyjay/'

    sample_dir = base_dir + exp
    samples = sorted(list(set([int(s.split('_')[-1][:-4]) for s in os.listdir(sample_dir) if 'samples' in s])))[:100]
    print "num samples : %d" % (len(samples))
    print sample_dir

    mean_stddev = np.zeros((len(samples),2))
    preds = []
    for i, s in enumerate(samples):
        #with open(sample_dir + '/samples_rec_%d.npy'%s) as f:
        with open(sample_dir + '/samples_%d.npy'%s) as f:
            images = np.load(f)
        images = np.split(images, images.shape[0])
        images = [im.reshape(im.shape[1:]) for im in images]

        mean, stddev, pred = model.get_inception_score(images)
        if i % 10 == 0:
            print "---------- SCORE in %s ------------" % s
            print "%f, %f"%(mean, stddev)
        mean_stddev[i,0] = mean
        mean_stddev[i,1] = stddev
        preds.append(pred)

    max_idx = mean_stddev[:,0].argmax()
    maxs.append((exp, mean_stddev[max_idx,0], mean_stddev[max_idx,1], samples[max_idx]))
    print 'MAX = %f, at %d' % (mean_stddev[max_idx,0], samples[max_idx])

    with open(sample_dir+'/scores.npy', 'w') as f:
        np.save(f, mean_stddev)
    with open(sample_dir+'/predictions.npy', 'w') as f:
        np.save(f, preds)
print maxs

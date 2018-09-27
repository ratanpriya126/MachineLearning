# -- coding: utf-8 --
"""
Created on Tue Sep 18 21:15:46 2018

@author: User
"""
# print(_doc_)

from Bio import SeqIO
import os
import numpy as np

from sklearn.metrics import euclidean_distances
from sklearn import manifold
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

pathToFile = os.path.join('HW2.fas')  #windows machine

allSeqs = []
for seq_record in SeqIO.parse(pathToFile, """fasta"""):
        allSeqs.append(seq_record.seq)

seqMat = np.array(allSeqs)
#print(seqMat)

as_int = seqMat.view(np.uint)
#print(as_int.shape)

X = np.array(as_int).astype(np.float)
#X_true.astype(np.float)

X.astype(np.float)

n=3751
seed = np.random.RandomState(seed=3)

X = X.reshape((n, 2))
#print(X)

X -= X.mean()
#print(X)

similarities = euclidean_distances(X)
#print(similarities)

mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=seed,
                   dissimilarity="precomputed", n_jobs=1)
pos = mds.fit(similarities).embedding_

# Rescale the data
pos = np.sqrt((X * 2).sum()) / np.sqrt((pos ** 2).sum())

# Rotate the data
clf = PCA(n_components=2)
X = clf.fit_transform(X)

pos = clf.fit_transform(pos)

fig = plt.figure(1)
ax = plt.axes([0., 0., 1., 1.])

s = 100
plt.scatter(X[:, 0], X[:, 1], color='navy', s=s, lw=0,
            label='True Position')
plt.scatter(pos[:, 0], pos[:, 1], color='turquoise', s=s, lw=0, label='MDS')
plt.legend(scatterpoints=1, loc='best', shadow=False)

pos = clf.fit_transform(pos)

fig = plt.figure(1)
ax = plt.axes([0., 0., 1., 1.])

s = 100
plt.scatter(X[:, 0], X[:, 1], color='navy', s=s, lw=0,
            label='True Position')
plt.scatter(pos[:, 0], pos[:, 1], color='turquoise', s=s, lw=0, label='MDS')
plt.legend(scatterpoints=1, loc='best', shadow=False)

# Plot the edges
#start_idx, end_idx = np.where(pos)

#segments = [[X[i, :], X[j, :]]
 #           for i in range(len(pos)) for j in range(len(pos))]
#values = np.abs(similarities)
#lc = LineCollection(segments,zorder=0, cmap=plt.cm.Blues,norm=plt.Normalize(0,values.max()))
#lc.set_array(similarities.flatten())
#lc.set_linewidths(0.5 * np.ones(len(segments)))
#ax.add_collection(lc)


plt.show()

print(pos)
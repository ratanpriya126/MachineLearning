#!/usr/bin/python 
from __future__ import division
import numpy as np
#import Bio
from Bio import SeqIO
from numpy import array
from numpy import mean
from numpy import cov
import numpy.linalg as linalg
from matplotlib import pyplot as plt
from sklearn.manifold import MDS

#Loading file and converting sequence into ASCII
mylist = list()
for record in SeqIO.parse("HW2.fas", "fasta"):
	mylist.append(record.seq)
	print(record.seq)
	myarray = np.asarray(mylist) 
	print(myarray.shape) 

# define a matrix
X_std = myarray.view(np.uint8)
print(X_std.shape)

#Principal Component Analysis
mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
print('Covariance matrix \n%s' %cov_mat)

eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)

for ev in eig_vecs.T:
    np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))

#Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])

matrix_w = eig_vecs[:,:2]

print(matrix_w.shape)
print('Matrix W:\n', matrix_w)

#Final Matrix
Y = X_std.dot(matrix_w)
print(Y.shape)
feature_vector1 = Y[:,0]
feature_vector2 = Y[:,1]
print(feature_vector1)
print(feature_vector1.shape)
print(feature_vector2)
print(feature_vector2.shape)
plt.xlabel('feature_vector1')
plt.ylabel('feature_vector2')
plt.title('Principal Component Analysis')
plt.grid('True')
plt.plot(feature_vector1,feature_vector2,'ro')
plt.show()

#Multidimentional Scaling
def hamdist(str1, str2):
	diffs = 0	
	for k in xrange(len(str1)):   
		if str1[k] != str2[k]:
			diffs += 1
	return diffs

rowscolumn = 62
MDS_Matrix = np.zeros((62, 62))

for i in xrange(rowscolumn):
	for j in xrange(rowscolumn):
		MDS_Matrix[i][j] = hamdist(X_std[i,:], X_std[j,:])
		
print(MDS_Matrix)

model = MDS(n_components=2, dissimilarity='precomputed', random_state=6)
out = model.fit_transform(MDS_Matrix)
print(out)
plt.title('MultiDimentional Scaling')
plt.xlabel('feature_vector1')
plt.ylabel('feature_vector2')
plt.scatter(out[:, 0], out[:, 1])
plt.show()

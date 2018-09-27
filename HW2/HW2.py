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
X_Matrix = myarray.view(np.uint8)
print(X_Matrix.shape)

#Principal Component Analysis
mean_vector = np.mean(X_Matrix, axis=0)
cov_matrix = (X_Matrix - mean_vector).T.dot((X_Matrix - mean_vector)) / (X_Matrix.shape[0]-1)
print('Covariance matrix \n%s' %cov_matrix)

eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

print('Eigenvectors \n%s' %eigenvectors)
print('\nEigenvalues \n%s' %eigenvalues)

for ev in eigenvectors.T:
    np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))

#Listing (eigenvalue, eigenvector) tuples
eigenpairs = [(np.abs(eigenvalues[i]), eigenvectors[:,i]) for i in range(len(eigenvalues))]

# Sorting Eigen values and Eigen Vectors
eigenpairs.sort(key=lambda x: x[0], reverse=True)

print('Eigenvalues in descending order:')
for i in eigenpairs:
    print(i[0])

matrix_w = eigenvectors[:,:2]

print(matrix_w.shape)
print('Matrix W:\n', matrix_w)

#Final Matrix
Y = X_Matrix.dot(matrix_w)
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
def findhamdist(str1, str2):
	diffs = 0	
	for k in xrange(len(str1)):   
		if str1[k] != str2[k]:
			diffs += 1
	return diffs

rowscolumn = 62
MDS_Matrix = np.zeros((62, 62))

for i in xrange(rowscolumn):
	for j in xrange(rowscolumn):
		MDS_Matrix[i][j] = findhamdist(X_Matrix[i,:], X_Matrix[j,:])
		
print(MDS_Matrix)

model = MDS(n_components=2, dissimilarity='precomputed', random_state=6)
out = model.fit_transform(MDS_Matrix)
print(out)
plt.title('MultiDimentional Scaling')
plt.xlabel('feature_vector1')
plt.ylabel('feature_vector2')
plt.scatter(out[:, 0], out[:, 1])
plt.show()

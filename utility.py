import numpy as np
import os

SCALE = 1000000000

def secure_aggregate_laplace(vectors, noise_scale, useMPC=False):
	if useMPC == False:
		return np.mean(vectors, axis=0) + noise_scale * np.random.laplace(0, 1, vectors.shape[1])

	vectors = (vectors*SCALE).astype(int)
	vectors_mask = np.random.randint(-SCALE, SCALE, vectors.shape)
	noise = (noise*SCALE).astype(int)
	noise_mask = np.random.randint(-SCALE, SCALE, noise.shape)
	fp1 = open('Inputs/beta1.txt', 'w')
	fp2 = open('Inputs/beta2.txt', 'w')
	fp3 = open('Inputs/noise1.txt', 'w')
	fp4 = open('Inputs/noise2.txt', 'w')
	for i in range(vectors.shape[0]):
		for j in range(vectors.shape[1]):
			fp1.write(str(vectors[i,j]^vectors_mask[i,j])+ ' ')
			fp2.write(str(vectors_mask[i,j])+ ' ')
		fp3.write(str(noise[i]^noise_mask[i])+ ' ')
		fp4.write(str(noise_mask[i])+ ' ')
	fp1.close()
	fp2.close()
	fp3.close()
	fp4.close()

	port = 1234

  	# Note: Currently, M, D, lambda, epsilon and chunk size for each party are all hard coded in modelAggregate.c and modelAggregate.oh files.
	os.system("./cycle './a.out "+str(port)+" -- dualex | ./a.out "+str(port)+" localhost dualex'")
	
	fp = open('Output/beta_avg.txt', 'r')
	beta = []
	for line in fp:
		beta = [float(val) for val in line.split(' ')[:-1]]
	fp.close()

	return np.array(beta)


def ratio_of_uniforms():
	u1 = np.random.rand()
	v2 = np.random.rand()
	u2 = (2*v2 - 1) * np.sqrt(2*np.exp(-1))
	x = u2 / u1
	while x*x > -4 * np.log(u1):
		u1 = np.random.rand()
		v2 = np.random.rand()
		u2 = (2*v2 - 1) * np.sqrt(2*np.exp(-1))
		x = u2 / u1
	return x


def secure_aggregate_gaussian(vectors, noise_scale, useMPC=False):
	if useMPC == False:
		return np.mean(vectors, axis=0) + noise_scale * np.random.normal(0, 1, vectors.shape[1])
		#return np.mean(vectors, axis=0) + noise_scale * [ratio_of_uniforms() for i in np.arange(vectors.shape[1])]

	vectors = (vectors*SCALE).astype(int)
	vectors_mask = np.random.randint(-SCALE, SCALE, vectors.shape)
	noise_scale = int(noise_scale*SCALE)
	noise_scale_mask = np.random.randint(-SCALE, SCALE)
	fp1 = open('Inputs/beta1.txt', 'w')
	fp2 = open('Inputs/beta2.txt', 'w')
	fp3 = open('Inputs/noise1.txt', 'w')
	fp4 = open('Inputs/noise2.txt', 'w')
	for i in range(vectors.shape[0]):
		for j in range(vectors.shape[1]):
			fp1.write(str(vectors[i,j]^vectors_mask[i,j])+ ' ')
			fp2.write(str(vectors_mask[i,j])+ ' ')
	fp3.write(str(noise_scale^noise_scale_mask)+ ' ')
	fp4.write(str(noise_scale_mask)+ ' ')
	fp1.close()
	fp2.close()
	fp3.close()
	fp4.close()

	port = 1234
	ret = 1

	while ret != 0:
		random1 = (np.random.rand(vectors.shape[1])*SCALE).astype(int)
		random2 = (np.random.rand(vectors.shape[1])*SCALE).astype(int)
		mask1 = np.random.randint(-SCALE, SCALE, vectors.shape[1])
		mask2 = np.random.randint(-SCALE, SCALE, vectors.shape[1])
		fp1 = open('Inputs/random11.txt', 'w')
		fp2 = open('Inputs/random12.txt', 'w')
		fp3 = open('Inputs/random21.txt', 'w')
		fp4 = open('Inputs/random22.txt', 'w')
		for i in range(vectors.shape[1]):
			fp1.write(str(random1[i]^mask1[i])+ ' ')
			fp2.write(str(mask1[i])+ ' ')
			fp3.write(str(random2[i]^mask2[i])+ ' ')
			fp4.write(str(mask2[i])+ ' ')
		fp1.close()
		fp2.close()
		fp3.close()
		fp4.close()

		ret = os.system("./cycle './a.out "+str(port)+" -- dualex "+str(vectors.shape[0])+" "+str(vectors.shape[1])+" | ./a.out "+str(port)+" localhost dualex "+str(vectors.shape[0])+" "+str(vectors.shape[1])+"'")
		
	fp = open('Output/beta_avg.txt', 'r')
	beta = []
	for line in fp:
		beta = [float(val) for val in line.split(' ')[:-1]]
	fp.close()

	return np.array(beta)

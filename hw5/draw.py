import numpy as np 
from scipy.special import softmax
import matplotlib.pyplot as plt
import sys
#input a nunpy array
#pred = model(img)
#a_array = pred.cpu().detach().numpy()
#np.save('a_array.npy', a_array)

prob_in = np.load(sys.argv[1])
prob_in = softmax(prob_in[0])

class picture :
	def __init__ (self,prob,index) :
		self.prob = prob
		self.index = index

probs = []

for i in range(len(prob_in)) :
	probs.append(picture(prob_in[i],i))
	

probs = sorted(probs,reverse=True,key = lambda p:p.prob)

x_ = [ str(e.index) for e in probs[0:3] ]
y_ = [ e.prob for e in probs[0:3] ]

print (x_)
print (y_)

plt.bar(x_,y_)
plt.show()





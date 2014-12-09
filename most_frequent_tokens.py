import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

def plot(tokens, cats=True):
	plt.figure(figsize=(1,1))
	xcats = tokens["word"].values
	x = range(len(xcats))
	y = tokens["words_global"].values
	plt.plot(x, y)
	if(cats):
		plt.xticks(x, xcats, size='small', rotation='vertical')
	plt.show()

if __name__ == "__main__":
	tokens = pd.read_csv("tokens.txt", sep="\t", header=None, names=['word','words_global','words_films'])
	sorted = tokens.sort(columns=['words_global'],ascending=False)
	print sorted
	#import pdb; pdb.set_trace()
	plot(sorted[0:100])
	plot(sorted, cats=False)
	

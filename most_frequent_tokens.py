import pandas as pd

if __name__ == "__main__":
	tokens = pd.read_csv("tokens.txt", sep="\t", header=None, names=['word','words_global','words_films'])
	print tokens.sort(columns=['words_global'],ascending=False)

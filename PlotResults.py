import cPickle

import numpy             as np
import numpy.random      as rd
import scipy.stats       as st
import scipy.special     as sp
import matplotlib.pyplot as plt

# Results with K = 10 and using movie scores:

results = np.array([
[
[26, 6869.635692, 756.068834],
[28, 6472.500550, 762.198183],
],
[
[26, 8165.890196, 932.947602],
[28, 6543.915124, 768.016767],
],
[
[26, 8220.925001, 798.305911],
[28, 6852.387612, 838.789814],
],
[
[26, 6430.593165, 802.857901],
[28, 7559.923453, 737.308147],
],
[
[26, 6405.251508, 636.769453],
[28, 6983.299462, 830.017438],
],
[
[26, 7068.113012, 734.310707],
[28, 8205.621880, 733.124837],
],
[
[26, 6453.213841, 809.770268],
[28, 6929.182339, 809.782729],
],
[
[26, 6760.742584, 896.918603],
[28, 7563.840973, 905.713510],
],
])

plt.figure(figsize = (16.0 / 2, 9.0 / 2))
plt.title("Perplexity")
plt.hold(True)
for c in range(results.shape[0]):
    plt.plot(results[c, :, 0].flatten(), results[c, :, 1].flatten(), color = "black")
plt.plot(results[c, :, 0].flatten(), np.mean(results[:, :, 1], axis = 0).flatten(), color = "black", linewidth = 5)
plt.savefig("results_perplexity.png")

plt.figure(figsize = (16.0 / 2, 9.0 / 2))
plt.title("Inverse accuracy")
plt.hold(True)
for c in range(results.shape[0]):
    plt.plot(results[c, :, 0].flatten(), results[c, :, 2].flatten(), color = "black")
plt.plot(results[c, :, 0].flatten(), np.mean(results[:, :, 2], axis = 0).flatten(), color = "black", linewidth = 5)
plt.savefig("results_accuracy.png")

#plt.show()
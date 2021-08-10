import numpy as np
import matplotlib.pyplot as plt

nsamples = 1000
effs_nn = np.random.rand(nsamples)
effs_bdt = effs_nn + np.random.uniform(-0.1, 0.1, nsamples)
effs_nn_inf = effs_nn + np.random.uniform(-.1, 0.01, nsamples)

fig, ax = plt.subplots(1,1)

violins = [effs_bdt - effs_nn, effs_bdt-effs_nn_inf]

ax.violinplot(violins, vert=False, showextrema=True, showmedians=True)
for y, xs in enumerate(violins):
  ax.scatter(xs, [y+1]*len(xs))
ax.set_yticks(range(1,len(violins) + 1))
ax.set_yticklabels(["bdt", "nn-inf"])
ax.set_xlabel("efficiency difference ($\epsilon_{\mathrm{unconstrained}} - \epsilon_{x}$)")
plt.show()

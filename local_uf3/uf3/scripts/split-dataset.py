from ase.io import read, write
import numpy as np

ims = read('inter-dataset.xyz', index=":", format="extxyz")
natl1 = 6
at_val = []
at_train = []
nsamp = 12
for ct, at in enumerate(ims):
    z_pos = at.get_positions()[:, 2]
    top_l1 = np.argmax(z_pos[:natl1])
    bot_l2 = np.argmin(z_pos[natl1:])
    dist = -at.get_positions()[top_l1, :] + at.get_positions()[natl1 + bot_l2, :]
    cell = at.cell
    dist[2] = 0

    sdist = np.array(np.round(cell.scaled_positions(dist) * nsamp), dtype=int)[:2]
    if sdist[0] % 2 == 0 and sdist[1] % 2 == 0:
        at_train.append(at)
    else:
        at_val.append(at)
    pass

train_file = "inter-training-refined.xyz"
test_file = "inter-test-refined.xyz"

write(train_file, at_train, format="extxyz")
write(test_file, at_val, format="extxyz")
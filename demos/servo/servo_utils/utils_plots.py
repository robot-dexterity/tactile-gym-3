import os
import numpy as np
import matplotlib.pylab as plt

from cri.transforms import quat2euler, euler2quat, inv_transform


class Contour3DPlotter:
    def __init__(self,
                 save_dir=None,
                 robot=None,
                 save_num=1,
                 name="contour_plot.png", 
                 limits=[[-110, 10], [-60, 60], [-30, 30]],
                 azim = 180,
                 inv=1,
        ):

        self.save_dir = save_dir
        self.save_num = save_num
        self.name = name

        self.counter = 0
        self.inv = inv
        self.v = [0, 0, 0, 0, 0, 0]

        if robot:
            azim = robot.workframe[5]
                
        plt.ion
        self._fig = plt.figure(name, figsize=(5, 5))
        self._fig.subplots_adjust(left=-0.1, right=1.1, bottom=-0.05, top=1.05)
        self._ax = self._fig.add_subplot(111, projection='3d')
        self._ax.view_init(30, 45, 15, 'z')
        self._ax.azim = azim
        self._ax.plot(limits[0], limits[1], limits[2], ':w')


    def update(self, v):
        self.v = np.vstack([self.v, v])
        self.counter += 1

        v_q = euler2quat([0, 0, 0, *self.v[-1, 3:]], axes='rxyz')
        d_q = euler2quat([-1/np.sqrt(2), 1/np.sqrt(2), 0, 0, 0, 0], axes='rxyz')
        w_q = euler2quat([0, 0, -1, 0, 0, 0], axes='rxyz')
        d = 5*quat2euler(inv_transform(d_q, v_q), axes='rxyz')
        w = 5*quat2euler(inv_transform(w_q, v_q), axes='rxyz')

        self._ax.plot(
            self.inv*self.v[-2:, 0], -self.v[-2:, 1], -self.v[-2:, 2],
            '-r')
        self._ax.plot(
            self.inv*self.v[-2:, 0]+[d[0], -d[0]], -self.v[-2:, 1]-[d[1], -d[1]], -self.v[-2:, 2]-[d[2], -d[2]],
            '-b', linewidth=0.5)
        self._ax.plot(
            self.inv*self.v[-2:, 0]+[w[0], 0], -self.v[-2:, 1]-[w[1], 0], -self.v[-2:, 2]-[w[2], 0],
            '-g', linewidth=0.5)

        save_now = self.counter % self.save_num == 0
        if save_now and self.save_dir is not None:
            save_file = os.path.join(self.save_dir, self.name)
            self._fig.savefig(save_file, dpi=320, pad_inches=0.01, bbox_inches='tight')


if __name__ == '__main__':
    pass

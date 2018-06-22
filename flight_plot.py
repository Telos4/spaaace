import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.widgets import TextBox

from data import PlanetData

class FlightPlot:
    def __init__(self, Ts, Xs):
        self.Ts = Ts
        self.Xs = Xs

        self.pd = PlanetData('GEO')

        self.box_lbx = min(min(Xs[:,0]), min(Xs[:,4]), min(Xs[:,8]))
        self.box_ubx = max(max(Xs[:,0]), max(Xs[:,4]), max(Xs[:,8]))
        self.box_lby = min(min(Xs[:,1]), min(Xs[:,5]), max(Xs[:,9]))
        self.box_uby = max(max(Xs[:,1]), max(Xs[:,5]), max(Xs[:,9]))
        self.box_border = max(self.box_ubx - self.box_lbx, self.box_uby - self.box_lby)/10.0
        self.box_lbx -= self.box_border
        self.box_lby -= self.box_border
        self.box_ubx += self.box_border
        self.box_uby += self.box_border

        pass

    def submit(self):
        print("submit!")

    def plot(self):
        self.fig, axs = plt.subplots(3,2)
        self.ax = axs[0][0]
        self.ax.axis('equal')

        text_box = TextBox(axs[1,1], 'Test Textbox', initial='initial_text')
        text_box.on_submit(self.submit)

        self.rocket_x = self.Xs[:,0]
        self.rocket_y = self.Xs[:,1]
        self.line_rocket, = self.ax.plot(self.rocket_x, self.rocket_y, 'k')

        earth_trajectory = self.pd.orb_E(self.Ts)
        self.earth_x = self.Xs[:,4]
        self.earth_y = self.Xs[:,5]
        self.line_earth, = self.ax.plot(self.earth_x, self.earth_y, color='b', linestyle='--')

        #self.earth = Ellipse((self.earth_x[-1], self.earth_y[-1]), 0.1 * self.pd.AE, 0.1 * self.pd.AE, color='blue')#self.pd.radius_earth, self.pd.radius_earth)
        self.earth = Ellipse((self.earth_x[-1], self.earth_y[-1]), self.pd.radius_earth, self.pd.radius_earth, color='blue')#self.pd.radius_earth, self.pd.radius_earth)
        self.ax.add_artist(self.earth)

        #mars_trajectory = self.pd.orb_M(self.Ts)
        self.mars_x = self.Xs[:,8]
        self.mars_y = self.Xs[:,9]
        self.line_mars, = self.ax.plot(self.mars_x, self.mars_y, color='r', linestyle='--')

        self.mars = Ellipse((self.mars_x[-1], self.mars_y[-1]), 0.1 * self.pd.AE, 0.1 * self.pd.AE, color='red')
        self.ax.add_artist(self.mars)

        self.sun_x = self.Xs[:,12]
        self.sun_y = self.Xs[:,13]
        self.line_sun, = self.ax.plot(self.sun_x, self.sun_y, color='y', linestyle='--')
        sun = Ellipse((self.sun_x[-1], self.sun_y[-1]),0.2*self.pd.AE,0.2*self.pd.AE, color='yellow')
        self.ax.add_artist(sun)

        self.text_rocket = self.ax.text(self.rocket_x[-1], self.rocket_x[-1], 'Rakete')
        self.text_earth = self.ax.text(self.earth_x[-1], self.earth_y[-1], 'Erde')
        self.text_mars = self.ax.text(self.mars_x[-1], self.mars_y[-1], 'Mars')
        self.text_sun = self.ax.text(self.sun_x[-1], self.sun_y[-1], 'Sonne')

        # set initial plot sector
        self.ax.set_ylim([self.box_lby, self.box_uby])
        self.ax.set_xlim([self.box_lbx, self.box_ubx])

        vel_ax = axs[0][1]
        self.rocket_vel_x = self.Xs[:,2]
        self.rocket_vel_y = self.Xs[:,3]
        vel = np.sqrt(self.rocket_vel_x**2 + self.rocket_vel_y**2)
        vel_ax.plot(self.Ts, vel)

        dist_ax1 = axs[1][0]
        dist_earth = np.linalg.norm(self.Xs[:,0:2] - self.Xs[:,4:6], axis=1)
        dist_ax1.plot(self.Ts, dist_earth)

        dist_ax2 = axs[2][0]
        dist_mars = np.linalg.norm(self.Xs[:,0:2] - self.Xs[:,8:10], axis=1)
        dist_ax2.plot(self.Ts, dist_mars)

        vel_ax2 = axs[2][1]
        vel_mars = np.linalg.norm(self.Xs[:,2:4] - self.Xs[:,10:12], axis=1)
        vel_ax2.plot(self.Ts, vel_mars)

        #return line_rocket, line_earth, line_mars, text_earth
        plt.show()

    def update(self, frame):
        print("frame {}".format(frame))
        print("p = ({},{})".format(self.Xs[frame,0], self.Xs[frame,1]))

        # position of rocket
        px = self.Xs[frame, 0]
        py = self.Xs[frame, 1]

        # update plot data for rocket
        self.rocket_x.append(px)
        self.rocket_y.append(py)
        self.line_rocket.set_data(self.rocket_x, self.rocket_y)

        # update plot data for planets
        pos_earth = self.pd.orb_E(self.Ts[frame])
        self.earth_x.append(pos_earth[0])
        self.earth_y.append(pos_earth[1])
        self.line_earth.set_data(self.earth_x, self.earth_y)
        self.earth.center = pos_earth

        pos_mars = self.pd.orb_M(self.Ts[frame])
        self.mars_x.append(pos_mars[0])
        self.mars_y.append(pos_mars[1])
        self.line_mars.set_data(self.mars_x, self.mars_y)
        self.mars.center = pos_mars

        # labels
        self.text_earth.x = pos_earth[0]
        self.text_earth.y = pos_earth[1]

        self.text_mars.x = pos_mars[0]
        self.text_mars.y = pos_mars[1]

        # update view
        self.box_lbx = min(self.box_lbx, px-self.box_size)
        self.box_ubx = max(self.box_ubx, px+self.box_size)
        self.box_lby = min(self.box_lby, py-self.box_size)
        self.box_uby = max(self.box_uby, py+self.box_size)
        self.ax.set_ylim([self.box_lby, self.box_uby])
        self.ax.set_xlim([self.box_lbx, self.box_ubx])
        self.ax.figure.canvas.draw() # redraw canvas (this is slow)

        # pause until next frame is ready to draw
        #pause = Ts[frame+1] - Ts[frame]
        #print("pause: {}".format(pause/1000.0))
        #time.sleep(pause/5000.0)

        return self.line_rocket, self.line_earth, self.line_mars, self.text_earth

    def animate(self):
        n = len(self.Ts)
        ani = anim.FuncAnimation(self.fig, self.update, frames=range(0, n), interval=40, blit=True,
                                 repeat=False)
        plt.show()
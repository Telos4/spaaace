import matplotlib.pyplot as plt
import matplotlib.animation as anim
from matplotlib.patches import Ellipse
import numpy as np

from data import PlanetData

class FlightAnim:
    def __init__(self, Ts, Xs):
        self.Ts = Ts
        self.Xs = Xs

        self.pd = PlanetData()

        max_x = np.max(np.abs([Xs[:,0], Xs[:,4], Xs[:,8]]))
        max_y = np.max(np.abs([Xs[:,1], Xs[:,5], Xs[:,9]]))
        max = np.max([max_x, max_y]) * 1.1

        self.box_lbx = -max
        self.box_ubx = max
        self.box_lby = -max
        self.box_uby = max
        #self.box_border = max(self.box_ubx - self.box_lbx, self.box_uby - self.box_lby)/10.0
        #self.box_lbx -= self.box_border
        #self.box_lby -= self.box_border
        #self.box_ubx += self.box_border
        #self.box_uby += self.box_border

        self.fig, self.ax = plt.subplots()

        self.ax.set_ylim([self.box_lby, self.box_uby])
        self.ax.set_xlim([self.box_lbx, self.box_ubx])
        self.ax.figure.canvas.draw()

        self.rocket_x = []
        self.rocket_y = []
        self.line_rocket, = self.ax.plot(self.rocket_x, self.rocket_y, 'k', animated=False)

        self.earth_x = []
        self.earth_y = []
        self.line_earth, = self.ax.plot(self.earth_x, self.earth_y, color='b', linestyle='--', animated=False)
        self.earth = Ellipse((1 * self.pd.AE, 0), 0.1 * self.pd.AE, 0.1 * self.pd.AE, animated=False, color='blue')

        self.mars_x = []
        self.mars_y = []
        self.line_mars, = self.ax.plot(self.mars_x, self.mars_y, color='r', linestyle='--', animated=False)
        self.mars = Ellipse((0, 1.5173 * self.pd.AE), 0.1 * self.pd.AE, 0.1 * self.pd.AE, animated=False, color='red')

        self.sun_x = []
        self.sun_y = []
        self.line_sun, = self.ax.plot(self.sun_x, self.sun_y, color='y', linestyle='--')
        self.sun = Ellipse((0,0), 0.2*self.pd.AE,0.2*self.pd.AE, animated=False, color='yellow')

        self.text_rocket = self.ax.text(0,0, 'Rakete', animated=False)
        self.text_earth = self.ax.text(0,0, 'Erde', animated=True)
        self.text_mars = self.ax.text(0,0, 'Mars')
        self.text_sun = self.ax.text(0,0, 'Sonne')
        self.text_time = self.ax.text(0.7*max, 0.9*max, 't = 0.0')

        # set initial plot sector
        #self.ax.axis('equal')
        #self.ax.set_ylim([self.box_lby, self.box_uby])
        #self.ax.set_xlim([self.box_lbx, self.box_ubx])
        self.ax.add_artist(self.earth)
        self.ax.add_artist(self.mars)
        self.ax.add_artist(self.sun)

        self.ax.add_artist(self.text_rocket)
        self.ax.add_artist(self.text_earth)
        self.ax.add_artist(self.text_time)

        self.ax.set_aspect('equal')
        #plt.show()
        pass
        #return line_rocket, line_earth, line_mars, text_earth

    def update(self, frame):
        print("frame {}".format(frame))
        print("p = ({},{})".format(self.Xs[frame,0], self.Xs[frame,1]))

        # position of rocket
        pos_rocket = self.Xs[frame, 0:2]

        print("Rocket")
        # update plot data for rocket
        self.rocket_x.append(pos_rocket[0])
        self.rocket_y.append(pos_rocket[1])
        self.line_rocket.set_data(self.rocket_x, self.rocket_y)

        print("Earth")
        # update plot data for planets
        pos_earth = self.Xs[frame, 4:6]
        self.earth_x.append(pos_earth[0])
        self.earth_y.append(pos_earth[1])
        self.line_earth.set_data(self.earth_x, self.earth_y)
        self.earth.center = pos_earth

        print("Mars")
        pos_mars = self.Xs[frame, 8:10]
        self.mars_x.append(pos_mars[0])
        self.mars_y.append(pos_mars[1])
        self.line_mars.set_data(self.mars_x, self.mars_y)
        self.mars.center = pos_mars

        print("Labels")
        # labels
        self.text_rocket.set_position(pos_rocket)
        self.text_earth.set_position(pos_earth)
        self.text_mars.set_position(pos_mars)
        #self.text_mars.y = pos_mars[1]

        self.text_time.set_text('t = {} days'.format(self.Ts[frame]/(60.0*60.0*24.0)))

        # update view
        #self.box_lbx = min(self.box_lbx, px-self.box_size)
        #self.box_ubx = max(self.box_ubx, px+self.box_size)
        #self.box_lby = min(self.box_lby, py-self.box_size)
        #self.box_uby = max(self.box_uby, py+self.box_size)
        #self.ax.set_ylim([self.box_lby, self.box_uby])
        #self.ax.set_xlim([self.box_lbx, self.box_ubx])
        #self.ax.figure.canvas.draw() # redraw canvas (this is slow)

        # pause until next frame is ready to draw
        #pause = Ts[frame+1] - Ts[frame]
        #print("pause: {}".format(pause/1000.0))
        #time.sleep(pause/5000.0)
        print("Done")

        return self.line_rocket, self.line_earth, self.line_mars, self.text_rocket, self.text_earth, self.text_mars, self.text_sun, self.text_time

    def animate(self):
        n = len(self.Ts)
        ani = anim.FuncAnimation(self.fig, self.update, frames=range(0, n, 200), interval=40, blit=False,
                                 repeat=False)
        plt.show()

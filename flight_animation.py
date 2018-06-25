import matplotlib.pyplot as plt
import matplotlib.animation as anim
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse
from matplotlib.widgets import TextBox, Button
from simulator import Simulator
import numpy as np
import ellipse_helpers

from flight_plot import FlightPlot

from data import PlanetData
import functools

class FlightAnim:
    def __init__(self):
        self.pd = PlanetData('GEO')
        self.simulator = Simulator(self.pd)

        self.tf = 1.0e8 # max time for simulation
        self.delta_t_mars = 0.0
        self.Xs = []
        self.Ts = []

        #max_x = np.max(np.abs([Xs[:,0], Xs[:,4], Xs[:,8]]))
        #max_y = np.max(np.abs([Xs[:,1], Xs[:,5], Xs[:,9]]))
        #max = np.max([max_x, max_y]) * 1.1
        max = 3.0 * self.pd.AE

        self.box_lbx = -max
        self.box_ubx = max
        self.box_lby = -max
        self.box_uby = max

        self.fig = plt.figure()
        gs = gridspec.GridSpec(10,10)
        self.ax = plt.subplot(gs[1:7, 0:6])
        command_input_fields = 5
        self.ax_command_input = []
        self.command_text_boxes = []
        self.commands = {0: np.array([0.0, 600.0, 0.0, 10.0])}
        for i in range(1, command_input_fields):
            self.commands[i] = None

        # define command input field
        for i in range(0, command_input_fields):
            self.ax_command_input.append(plt.subplot(gs[i, 7:]))
            if i == 0:
                initial = str(self.commands[i][0]) + " " + str(self.commands[i][1]) + " " + str(self.commands[i][2]) \
                          + " " + str(self.commands[i][3])
            else:
                initial = ""
            self.command_text_boxes.append(TextBox(self.ax_command_input[i], 'Befehl {} '.format(i+1), initial=initial))
            f = functools.partial(self.submit_command, i)
            self.command_text_boxes[i].on_submit(f)

        self.ax_mars_delta_t = plt.subplot(gs[command_input_fields, 7:])
        self.delta_t_mars_text_box = TextBox(self.ax_mars_delta_t, 'Delta t Mars ', initial='0.0')
        self.delta_t_mars_text_box.on_submit(self.submit_delta_t_mars)


        # outputs
        self.ax_time = plt.subplot(gs[0, 4:6])
        self.ax_time.get_xaxis().set_visible(False)
        self.ax_time.get_yaxis().set_visible(False)
        self.text_time = self.ax_time.text(0.1,0.5, 't = 0.0')


        self.ax_vel_rocket = plt.subplot(gs[7, 0:3])
        self.ax_vel_rocket.get_xaxis().set_visible(False)
        self.ax_vel_rocket.get_yaxis().set_visible(False)
        self.text_vel_rocket = self.ax_vel_rocket.text(0.1,0.5, 'v_{R} = (0.0, 0.0)')

        self.ax_vel_mars = plt.subplot(gs[8, 0:3])
        self.ax_vel_mars.get_xaxis().set_visible(False)
        self.ax_vel_mars.get_yaxis().set_visible(False)
        self.text_vel_mars = self.ax_vel_mars.text(0.1, 0.5, 'v_{M} = (0.0, 0.0)')

        self.ax_button = plt.subplot(gs[9,9])

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
        self.text_earth = self.ax.text(0,0, 'Erde', animated=False, verticalalignment='top')
        self.text_mars = self.ax.text(0,0, 'Mars')
        self.text_sun = self.ax.text(0,0, 'Sonne')

        # set initial plot sector
        #self.ax.axis('equal')
        #self.ax.set_ylim([self.box_lby, self.box_uby])
        #self.ax.set_xlim([self.box_lbx, self.box_ubx])
        self.ax.add_artist(self.earth)
        self.ax.add_artist(self.mars)
        self.ax.add_artist(self.sun)

        self.ax.add_artist(self.text_rocket)
        self.ax.add_artist(self.text_earth)

        self.ax_time.add_artist(self.text_time)

        self.button_run = Button(self.ax_button, 'Start')
        self.button_run.on_clicked(self.callback_button_run)

        self.fig.canvas.mpl_connect('button_press_event', self.onClick)
        self.fig.canvas.mpl_connect('key_press_event', self.handleKeys)

        self.ax.set_aspect('equal')
        #plt.show()

        self.speed = 10
        self.myframe = 0

        self.anim_running = False

        pass
        #return line_rocket, line_earth, line_mars, text_earth

    def submit_command(self, i, text):
        # TODO check text
        print("submit function "+str(i))
        print("text = " + text)
        data = np.fromstring(text, sep=' ')
        if self.check_data(data, i) == True:
            print("good data")
            print("data = {}".format(data))
            self.commands[i] = data
        else:
            print("invalid data")
            self.commands[i] = None

    def check_data(self, data, i):
        if len(data) != 4:
            return False
        if data[0] < 0 or data[1] <= 0.0:
            return False
        # check if time point is too early
        for j in range(0, i):
            if self.commands[j] is not None:
                if self.commands[j][0] + self.commands[j][1] > data[0]:
                    return False
        # check if duration is too long
        for j in range(i+1, len(self.commands.keys())):
            if self.commands[j] is not None:
                if data[0] + data[1] > self.commands[j][0]:
                    return False
        return True

    def submit_delta_t_mars(self, text):
        # TODO check text
        self.delta_t_mars = float(text)

    def callback_button_run(self, event):
        self.reset()
        print("Button pressed -> rerun simulation")

    def update(self, frame):
        #print("  frame {}".format(frame))
        #print("myframe {}".format(self.myframe))
        self.myframe = min(self.myframe, len(self.Ts)-1)
        #print("p = ({},{})".format(self.Xs[frame,0], self.Xs[frame,1]))

        # position of rocket
        pos_rocket = self.Xs[self.myframe, 0:2]

        # update plot data for rocket
        self.rocket_x.append(pos_rocket[0])
        self.rocket_y.append(pos_rocket[1])
        self.line_rocket.set_data(self.rocket_x, self.rocket_y)

        # update plot data for planets
        pos_earth = self.Xs[self.myframe, 4:6]
        self.earth_x.append(pos_earth[0])
        self.earth_y.append(pos_earth[1])
        self.line_earth.set_data(self.earth_x, self.earth_y)
        self.earth.center = pos_earth

        pos_mars = self.Xs[self.myframe, 8:10]
        self.mars_x.append(pos_mars[0])
        self.mars_y.append(pos_mars[1])
        self.line_mars.set_data(self.mars_x, self.mars_y)
        self.mars.center = pos_mars

        # labels
        self.text_rocket.set_position(pos_rocket)
        self.text_earth.set_position(pos_earth)
        self.text_mars.set_position(pos_mars)

        self.text_time.set_text('t = {:6.0f} s'.format(self.Ts[self.myframe]))
        self.text_vel_rocket.set_text('v_R = ({:6.0f}, {:6.0f})'.format(self.Xs[self.myframe][2], self.Xs[self.myframe][3]))
        self.text_vel_mars.set_text('v_M = ({:6.0f}, {:6.0f})'.format(self.Xs[self.myframe][10], self.Xs[self.myframe][11]))

        self.myframe += max(1, self.speed*self.speed)

        return self.line_rocket, self.line_earth, self.line_mars, self.text_rocket, self.text_earth, self.text_mars, \
               self.text_sun, self.text_time, self.command_text_boxes


    def onClick(self, event):
        print("Click Event!")


    def handleKeys(self, event):
        #print("Key pressed!")
        if event.key == 'down':
            if self.anim_running:
                self.anim.event_source.stop()
                self.anim_running = False
            else:
                self.anim.event_source.start()
                self.anim_running = True
            print("Pause")

        # control simulation speed
        if event.key == '-':
            self.speed -= 1
            self.speed = max(self.speed, 1)
            print("speed = {}".format(self.speed))
        elif event.key == '+':
            self.speed += 1
            print("speed = {}".format(self.speed))
        elif event.key == 's':
            print("autosolve")

    def auto_solve(self):
        # reset to get a clean state
        if self.commands[0] is not None:
            for i in range(1, len(self.commands.keys())):
                self.commands[i] = None
            self.delta_t_mars = 0.0
            self.reset()

        # TODO set correct bounds
        samples_r = self.Xs[2000:-1:5000, 0:2]
        samples_m = self.Xs[2000:-1:5000, 8:10]

        scaling = np.max([np.abs(samples_r), np.abs(samples_m)])
        samples_r_scaled = samples_r / scaling
        samples_m_scaled = samples_m / scaling

        # ellipse equation: a x^2 + b xy + c y^2 + d x + e y + f = 0
        er_params_scaled, er_center_scaled, er_angle_scaled, er_axis_scaled = ellipse_helpers.compute_ellipse_params(samples_r_scaled)
        em_params_scaled, em_center_scaled, em_angle_scaled, em_axis_scaled = ellipse_helpers.compute_ellipse_params(samples_m_scaled)

        sol_scaled = ellipse_helpers.ellipse_intersection(er_params_scaled, em_params_scaled, self.Xs[0, 0:2] / scaling)
        sol = sol_scaled * scaling

        # find out intersection time of rocket with point
        T_intersect_rocket_index = np.argmin(np.linalg.norm(self.Xs[:, 0:2] - sol, axis=1))
        T_intersect_rocket = self.Ts[T_intersect_rocket_index]

        # find out intersection time of rocket with point
        T_intersect_mars_index = np.argmin(np.linalg.norm(self.Xs[:, 8:10] - sol, axis=1))
        T_intersect_mars = self.Ts[T_intersect_mars_index]

        self.delta_t_mars = T_intersect_mars - T_intersect_rocket

        vel_rocket = self.Xs[T_intersect_rocket_index, 2:4]
        vel_mars = self.Xs[T_intersect_mars_index, 10:12]
        delta_v = vel_mars - vel_rocket
        delta_v_norm = delta_v / np.linalg.norm(delta_v)
        e1 = np.array([1.0, 0.0])
        w_des = np.arccos(np.dot(delta_v_norm, e1))
        if delta_v_norm[1] < 0.0:
            # if vector points down in the y-component we have to adjust the angle such that it is mathematically positive
            w_des = 2.0 * np.pi - w_des

        a_des = delta_v / 1200.0
        u_des = a_des[0] / np.cos(w_des)

        self.commands[1] = np.array([T_intersect_rocket, 1200.0, w_des, u_des])

        new_text = str(self.commands[1][0]) + " " + str(self.commands[1][1]) + " " + str(self.commands[1][2]) \
                  + " " + str(self.commands[1][3])
        self.command_text_boxes[1].set_val(new_text)

        self.delta_t_mars_text_box.set_val(str(self.delta_t_mars))

    def reset(self):
        if self.anim_running == True:
            self.anim.event_source.stop()

        self.rocket_x = []
        self.rocket_y = []

        self.mars_x = []
        self.mars_y = []

        self.earth_x = []
        self.earth_y = []

        if self.delta_t_mars != 0.0:
            print("running simulation for mars position")
            x_temp = self.simulator.run_simulation_planets(self.delta_t_mars)
            print("new position = {}".format(x_temp))
            self.simulator.set_mars_position(x_temp[4:8])
        else:
            print("resetting mars position")
            self.simulator.reset_mars_position()

        commands_clean = None
        for i in range(0, len(self.commands.keys())):
            print(commands_clean)
            print(self.commands[i])
            if self.commands[i] is not None:
                if commands_clean is None:
                    commands_clean = np.array([self.commands[i]])
                else:
                    commands_clean = np.vstack((commands_clean, self.commands[i]))

        self.Ts, self.Xs = self.simulator.run_simulation(commands_clean, self.tf)
        n = len(self.Ts)

        self.myframe = 0
        self.anim = anim.FuncAnimation(self.fig, self.update, frames=range(0, n, 1), interval=40, blit=False,
                                 repeat=False)
        self.anim_running = True

    def animate(self):
        n = len(self.Ts)
        self.anim = anim.FuncAnimation(self.fig, self.update, frames=range(0, n, 1), interval=40, blit=False,
                                 repeat=False)
        self.anim_running = True
        plt.show()

    def main_loop(self):
        while True:
            self.animate()
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse
from matplotlib.widgets import TextBox, Button
from simulator import Simulator
import numpy as np
import ellipse_helpers

from data import PlanetData
import functools


class FlightAnim:
    def __init__(self):
        self.pd = PlanetData('GEO')
        self.simulator = Simulator(self.pd)

        self.tf = 1.0e8  # max time for simulation
        self.delta_t_mars = 0.0
        self.Xs = []
        self.Ts = []

        self.difficulty = 1
        # 1 = just hit space around mars in distance of 0.1 AE
        # 2 = 0.01 AE
        # 3 = orbit

        max = 3.0 * self.pd.AE

        self.box_lbx = -max
        self.box_ubx = max
        self.box_lby = -max
        self.box_uby = max

        self.fig = plt.figure()
        gs = gridspec.GridSpec(10, 10)
        self.ax = plt.subplot(gs[1:7, 0:6])
        self.ax.grid(True)
        command_input_fields = 5
        self.ax_command_input = []
        self.command_text_boxes = []
        self.commands = {0: np.array([0.0, 600.0, 0.0, 10.0])}
        for i in range(1, command_input_fields):
            self.commands[i] = None

        # set up axes
        for i in range(0, len(self.commands.keys())):
            self.ax_command_input.append(plt.subplot(gs[i, 7:]))

        # define command input field
        for i in range(0, len(self.commands.keys())):
            if i == 0:
                initial = str(self.commands[i][0]) + " " + str(self.commands[i][1]) + " " + str(self.commands[i][2]) \
                          + " " + str(self.commands[i][3])
            else:
                initial = ""
            self.command_text_boxes.append(
                TextBox(self.ax_command_input[i], 'Befehl {} '.format(i + 1), initial=initial))
            f = functools.partial(self.submit_command, i)
            self.command_text_boxes[i].on_submit(f)

        self.ax_mars_delta_t = plt.subplot(gs[command_input_fields, 7:])

        self.delta_t_mars_text_box = TextBox(self.ax_mars_delta_t, 'Delta t Mars ', initial='0.0')
        self.delta_t_mars_text_box.on_submit(self.submit_delta_t_mars)

        # outputs
        self.ax_time = plt.subplot(gs[0, 4:6])
        self.ax_time.get_xaxis().set_visible(False)
        self.ax_time.get_yaxis().set_visible(False)
        self.text_time = self.ax_time.text(0.1, 0.5, 't = 0.0')

        self.ax_level = plt.subplot(gs[0, 0:2])
        self.ax_level .get_xaxis().set_visible(False)
        self.ax_level .get_yaxis().set_visible(False)
        self.text_level = self.ax_level.text(0.1, 0.5, 'Level 1')

        self.ax_info = plt.subplot(gs[7:10, 0:5])
        self.ax_info.get_xaxis().set_visible(False)
        self.ax_info.get_yaxis().set_visible(False)
        self.text_vel_rocket = self.ax_info.text(0.1, 0.25, 'v_{R} = (0.0, 0.0)')

        self.ax_info.get_xaxis().set_visible(False)
        self.ax_info.get_yaxis().set_visible(False)
        self.text_vel_mars = self.ax_info.text(0.1, 0.75, 'v_{M} = (0.0, 0.0)')

        self.victory_text = self.ax_info.text(0.1, 0.25, 'Victory!')

        self.ax_button = plt.subplot(gs[9, 9])

        self.ax.set_ylim([self.box_lby, self.box_uby])
        self.ax.set_xlim([self.box_lbx, self.box_ubx])
        self.ax.figure.canvas.draw()

        self.rocket_x = []
        self.rocket_y = []
        self.line_rocket, = self.ax.plot(self.rocket_x, self.rocket_y, 'k', animated=False)
        self.point_rocket, = self.ax.plot([], [], color='k', marker='o', markersize=2, animated=False)

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
        self.sun = Ellipse((0, 0), 0.2 * self.pd.AE, 0.2 * self.pd.AE, animated=False, color='yellow')

        self.text_rocket = self.ax.text(0, 0, 'Rakete', verticalalignment='bottom')
        self.text_earth = self.ax.text(0, 0, 'Erde', verticalalignment='top')
        self.text_mars = self.ax.text(0, 0, 'Mars', verticalalignment='top')
        self.text_sun = self.ax.text(0, 0, 'Sonne', verticalalignment='top')

        self.ax.add_artist(self.earth)
        self.ax.add_artist(self.mars)
        self.ax.add_artist(self.sun)

        self.ax.add_artist(self.text_rocket)
        self.ax.add_artist(self.text_earth)
        self.ax.add_artist(self.text_mars)
        self.ax.add_artist(self.text_sun)

        self.ax_time.add_artist(self.text_time)

        self.button_run = Button(self.ax_button, 'Start', color='g', hovercolor='g')
        self.button_run.on_clicked(self.callback_button_run)

        self.fig.canvas.mpl_connect('motion_notify_event', self.axes_enter)
        self.fig.canvas.mpl_connect('resize_event', self.axes_enter)
        #self.fig.canvas.mpl_connect('button_press_event', self.axes_enter)

        self.fig.canvas.mpl_connect('key_press_event', self.handleKeys)

        self.ax.set_aspect('equal')

        self.speed = 10
        self.myframe = 0

        self.anim_running = False
        self.anim = None
        self.redraw = False

    def axes_enter(self, event):
        if self.redraw:
            self.line_earth.axes.draw_artist(self.line_earth)
            self.line_mars.axes.draw_artist(self.line_mars)
            self.line_rocket.axes.draw_artist(self.line_rocket)
            self.point_rocket.axes.draw_artist(self.point_rocket)
            self.sun.axes.draw_artist(self.sun)
            self.earth.axes.draw_artist(self.earth)
            self.mars.axes.draw_artist(self.mars)
            self.text_sun.axes.draw_artist(self.text_sun)
            self.text_mars.axes.draw_artist(self.text_mars)
            self.text_earth.axes.draw_artist(self.text_earth)
            self.text_rocket.axes.draw_artist(self.text_rocket)
            self.text_time.axes.draw_artist(self.text_time)
            self.text_vel_rocket.axes.draw_artist(self.text_vel_rocket)
            self.text_vel_mars.axes.draw_artist(self.text_vel_mars)
            self.text_level.draw_artist(self.text_level)
            self.victory_text.axes.draw_artist(self.victory_text)
            self.ax.figure.canvas.blit(self.ax.bbox)
            self.ax_time.figure.canvas.blit(self.ax_time.bbox)
            self.ax_info.figure.canvas.blit(self.ax_info.bbox)

    def init_func(self):
        # initialize plot data
        self.line_rocket.set_data([], [])
        self.line_earth.set_data([], [])
        self.line_mars.set_data([], [])
        self.point_rocket.set_data([], [])

        self.earth.center = (0, 0)
        self.mars.center = (0, 0)
        self.sun.center = (0, 0)

        self.text_rocket.set_text('')
        self.text_earth.set_text('')
        self.text_mars.set_text('')
        self.text_sun.set_text('')
        self.text_time.set_text('')
        self.text_vel_rocket.set_text('')
        self.text_vel_mars.set_text('')
        self.victory_text.set_text('')
        self.text_level.set_text('')

        print("init!")

        return self.line_rocket, self.line_earth, self.line_mars, self.point_rocket, self.earth, self.mars, self.sun, \
               self.text_rocket, self.text_earth, self.text_mars, self.text_sun, self.text_time, self.text_vel_rocket, \
               self.text_vel_mars, self.victory_text, self.text_level

    def update_func(self, frame):
        self.myframe = min(self.myframe, len(self.Ts) - 1)

        self.text_rocket.set_text('Rakete')
        self.text_earth.set_text('Erde')
        self.text_mars.set_text('Mars')
        self.text_sun.set_text('Sonne')

        # update plot data for rocket
        pos_rocket = self.Xs[self.myframe, 0:2]
        self.rocket_x.append(pos_rocket[0])
        self.rocket_y.append(pos_rocket[1])
        self.line_rocket.set_data(self.rocket_x, self.rocket_y)
        self.point_rocket.set_data(pos_rocket[0], pos_rocket[1])
        self.text_rocket.set_position(pos_rocket)

        # update plot data for earth
        pos_earth = self.Xs[self.myframe, 4:6]
        self.earth_x.append(pos_earth[0])
        self.earth_y.append(pos_earth[1])
        self.line_earth.set_data(self.earth_x, self.earth_y)
        self.earth.center = pos_earth
        self.text_earth.set_position(pos_earth)

        # update plot data for mars
        pos_mars = self.Xs[self.myframe, 8:10]
        self.mars_x.append(pos_mars[0])
        self.mars_y.append(pos_mars[1])
        self.line_mars.set_data(self.mars_x, self.mars_y)
        self.mars.center = pos_mars
        self.text_mars.set_position(pos_mars)

        # update displayed texts
        t = self.Ts[self.myframe]
        vel_rocket = self.Xs[self.myframe][2:4]
        vel_mars = self.Xs[self.myframe][10:12]
        self.text_time.set_text('t = {:6.0f} s'.format(t))
        self.text_vel_rocket.set_text(
            'v_R = ({:6.0f}, {:6.0f})'.format(vel_rocket[0], vel_rocket[1]))
        self.text_vel_mars.set_text(
            'v_M = ({:6.0f}, {:6.0f})'.format(vel_mars[0], vel_mars[1]))
        self.text_level.set_text('Level {}'.format(self.difficulty))

        # check victory conditions
        dist_rocket_mars = np.linalg.norm(np.array(pos_rocket) - np.array(pos_mars))
        vel_dist_rocket_mars = np.linalg.norm(np.array(vel_rocket) - np.array(vel_mars))
        self.check_victory(dist_rocket_mars, vel_dist_rocket_mars)

        self.myframe += max(1, self.speed * self.speed)

        return self.line_rocket, self.line_earth, self.line_mars, self.point_rocket, self.earth, self.mars, self.sun, \
               self.text_rocket, self.text_earth, self.text_mars, self.text_sun, self.text_time, self.text_vel_rocket, \
               self.text_vel_mars, self.victory_text, self.text_level

    def check_victory(self, dist, vel_dist):
        victory = False
        if self.difficulty == 1:
            if dist < 0.05 * self.pd.AE:
                victory = True
        elif self.difficulty == 2:
            if dist < 0.005 * self.pd.AE:
                victory = True
        elif self.difficulty == 3:
            if dist <= 3.0e9 and vel_dist <= 1.0e3:
                victory = True
                print("Orbit achieved!")

        if victory == True:
            print("Victory!")
            # display information
            # time, energy, dist, delta_v
            self.text_vel_mars.set_text('')
            self.text_vel_rocket.set_text('')

            days = self.Ts[self.myframe]/(60.0*60.0*24.0)
            dist = dist/1000.0
            energy = 0.0
            for k in self.commands.keys():
                if self.commands[k] is not None:
                    energy += self.commands[k][1] * self.commands[k][3]**2
            text = 'Mars erreicht!\n\n Zeit (d): {:6.2f} \n Abstand (km): {:6.2f} \n Delta v (m/s): {:6.2f} \n ' \
                   'Treibstoffverbrauch: {:6.2f}'.format(days, dist, vel_dist, energy)
            self.victory_text.set_text(text)

            self.anim_running = False
            self.redraw = True
            self.anim.event_source.stop()

        else:
            self.victory_text.set_text('')


    def submit_command(self, i, text):
        # TODO check text
        print("submit function " + str(i))
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
        for j in range(i + 1, len(self.commands.keys())):
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

    def onClick(self, event):
        print("Click Event!")

    def handleKeys(self, event):
        if event.key == 'down':
            if self.anim_running:
                self.anim.event_source.stop()
                self.anim_running = False
                self.redraw = True
            else:
                self.anim.event_source.start()
                self.anim_running = True
                self.redraw = False
            print("Pause")

        # control simulation speed
        if event.key == '-':
            self.speed -= 1
            self.speed = max(self.speed, 1)
            print("speed = {}".format(self.speed))
        elif event.key == '+':
            self.speed += 1
            print("speed = {}".format(self.speed))
        elif event.key == 'ctrl+l':
            print("autosolve")
            self.auto_solve()
        elif event.key == 'ctrl+1':
            self.difficulty = 1
        elif event.key == 'ctrl+2':
            self.difficulty = 2
        elif event.key == 'ctrl+3':
            self.difficulty = 3

        if self.difficulty == 1:
            self.mars.width = 0.1 * self.pd.AE
            self.mars.height = 0.1 * self.pd.AE
            self.earth.width = 0.1 * self.pd.AE
            self.earth.height = 0.1 * self.pd.AE
        elif self.difficulty == 2:
            self.mars.width = 0.01 * self.pd.AE
            self.mars.height = 0.01 * self.pd.AE
            self.earth.width = 0.01 * self.pd.AE
            self.earth.height = 0.01 * self.pd.AE
        elif self.difficulty > 2:
            self.mars.width = 6800.0e3
            self.mars.height = 6800.0e3
            self.earth.width = 12742.0e3
            self.earth.height = 12742.0e3

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
        er_params_scaled, er_center_scaled, er_angle_scaled, er_axis_scaled = ellipse_helpers.compute_ellipse_params(
            samples_r_scaled)
        em_params_scaled, em_center_scaled, em_angle_scaled, em_axis_scaled = ellipse_helpers.compute_ellipse_params(
            samples_m_scaled)

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
        w_des = w_des * 360.0 / (2.0 * np.pi)

        self.commands[1] = np.array([T_intersect_rocket, 1200.0, w_des, u_des])

        new_text = "{:6.0f} {:6.2f} {:6.2f} {:6.2f}".format(self.commands[1][0],self.commands[1][1], self.commands[1][2],self.commands[1][3])
        self.command_text_boxes[1].set_val(new_text)

        new_time_mars = "{:6.0f} ".format(self.delta_t_mars)
        self.delta_t_mars_text_box.set_val(new_time_mars)

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

        commands_clean = self.clean_commands()
        self.Ts, self.Xs = self.simulator.run_simulation(commands_clean, self.tf)

        self.myframe = 0
        self.anim_running = True
        self.anim.event_source.start()

    def clean_commands(self):
        commands_clean = None
        for i in range(0, len(self.commands.keys())):
            print(commands_clean)
            print(self.commands[i])
            if self.commands[i] is not None:
                c = np.copy(self.commands[i])
                c[2] = c[2] / 360.0 * 2.0 * np.pi # convert to rad

                print("c = {}".format(c))
                if commands_clean is None:
                    commands_clean = np.array([c])
                else:
                    commands_clean = np.vstack((commands_clean, c))
        return commands_clean

    def main_loop(self):
        commands_clean = self.clean_commands()
        self.Ts, self.Xs = self.simulator.run_simulation(commands_clean, self.tf)

        self.anim_running = True
        self.anim = animation.FuncAnimation(self.fig, self.update_func, frames=100,
                                            interval=30, blit=True, init_func=self.init_func)
        plt.show()

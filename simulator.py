import odes
import numpy as np
import data
from scipy.integrate import solve_ivp
import time

class Simulator():
    def __init__(self, pd):
        self.t0 = 0.0
        self.tf = 1500000.0
        self.dt = 60.0

        self.x0_planets = np.zeros(12)

        # earth
        self.x0_planets[0] = pd.AE
        self.x0_planets[1] = 0
        self.x0_planets[2] = 0
        self.x0_planets[3] = pd.earth_orbital_speed

        # mars
        self.x0_planets[4] = pd.mars_trajectory_radius
        self.x0_planets[5] = 0
        self.x0_planets[6] = 0
        self.x0_planets[7] = pd.mars_orbital_speed

        # sun
        self.x0_planets[8] = 0
        self.x0_planets[9] = 0
        self.x0_planets[10] = 0
        self.x0_planets[11] = 0

        #self.x0_planets[4:8] = np.array([ 2.02983240e+11,  1.03711178e+11, -1.09818735e+04,  2.14969135e+04])

        r = pd.radius_earth + pd.initial_height

        self.x0 = np.zeros(16)
        self.x0[0] = pd.AE + r
        self.x0[1] = 0.0
        self.x0[2] = 0.0
        self.x0[3] = pd.earth_orbital_speed + pd.orbital_speed

        self.x0[4:16] = self.x0_planets[0:12]

        odes.event_mars_hit.terminal = False
        odes.event_timeout.terminal = False


    def run_simulation(self, commands, tf):
        x = self.x0
        t = self.t0
        t_timer_start = time.clock()
        odes.t_timer_start = t_timer_start
        Ts = np.array([t])
        Xs = np.array([x])

        # convert commands to format used by solver
        # commands = (time, duration, angle, thrust)
        # commands_mode = (time, angle, thrust)
        if commands[0,0] > t:
            commands_mod = [np.zeros(3)]
        else:
            commands_mod = []
        j = 0
        for i in range(0,len(commands[:,0])):
            c_mod = np.zeros(3)
            c_mod[0] = commands[i, 0] # time
            c_mod[1] = commands[i, 2] # angle
            c_mod[2] = commands[i, 3] # thrust
            commands_mod.append(c_mod)
            j += 1

            if i+1 < len(commands[:,0]): # not last control
                if commands[i, 0] + commands[i, 1] < commands[i+1, 0]: # check if next control is directly after this one
                    # if not add pause between controls
                    c_mod = np.zeros(3)
                    c_mod[0] = commands[i, 0] + commands[i, 1]
                    c_mod[1] = 0.0
                    c_mod[2] = 0.0
                    commands_mod.append(c_mod)
                    j += 1
            else: # last control
                if commands[-1, 0] + commands[-1, 1] < tf: # check if we have to add another pause
                    # if not add pause between controls
                    c_mod = np.zeros(3)
                    c_mod[0] = commands[i, 0] + commands[i, 1]
                    c_mod[1] = 0.0
                    c_mod[2] = 0.0
                    commands_mod.append(c_mod)
                    j += 1
                # add value for final time
                commands_mod.append(np.array([tf, 0.0, 0.0]))

        for i in range(0, len(commands_mod)-1):
            t0 = commands_mod[i][0]
            t1 = min(commands_mod[i+1][0], tf)
            w = commands_mod[i][1]
            u = commands_mod[i][2]

            t_eval = np.linspace(t0, t1, np.ceil((t1-t0)/self.dt)+1)

            sol = solve_ivp(fun= lambda t, x: odes.f_full(t, x, u, w), t_span=(t0, t1), y0=x, t_eval=t_eval, method='RK45',
                            rtol=1.0e-5, atol=1.0e-8, events=[odes.event_mars_hit, odes.event_timeout])
            Ts = np.concatenate((Ts, sol.t[1:]))
            Xs = np.concatenate((Xs, np.transpose(sol.y[:,1:])))
            x = Xs[-1]
            t = t1
            pass
        t_timer_end = time.clock()
        print("ODE solve duration: {}".format(t_timer_end-t_timer_start))

        return Ts, Xs
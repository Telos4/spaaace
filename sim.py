import numpy as np
from numpy.linalg import eig, inv
from scipy.integrate import ode, solve_ivp
from scipy.optimize import fsolve
from matplotlib.patches import Ellipse

from flight_animation import FlightAnim
from flight_plot import FlightPlot

#from odes import f_planets, f_full, event_mars_hit, event_timeout
import odes
from data import PlanetData

import simulator

pd = PlanetData('GEO')

if __name__ == "__main__":
    #simulator = simulator.Simulator(pd)
    #commands = np.array([[0.0, 1200.0, 0.0, 5.0], [3000.0, 100.0, 0.0, 0.0]])
    #Ts, Xs = simulator.run_simulation(commands, 1.0e8)

    #fp = FlightPlot(Ts, Xs)
    #fp.plot()

    fa = FlightAnim()
    #fa.auto_solve()
    fa.main_loop()

    #simulator = simulator.Simulator(pd)
    # commands = (time, duration, angle, thrust)
    #commands = np.array([[0.0, 1200.0, 0.0, 5.0],[13352221.0, 1200.0, 4.4057, 3.8938]])
    #commands = np.array([[110.0, 1200.0, 0.0, 5.0],[1500.0, 100.0, 4.4057, 3.8938]])

    #Ts, Xs = simulator.run_simulation(commands, tf)

    #fp = FlightPlot(Ts, Xs)
    #fp.plot()

    #fa = FlightAnim(Ts, Xs)
    #fa.animate()

    # compute ellipse equation by solving a linear system
    samples_r = Xs[2000:-1:5000,0:2]
    samples_m = Xs[2000:-1:5000,8:10]

    scaling = np.max([np.abs(samples_r), np.abs(samples_m)])
    samples_r_scaled = samples_r / scaling
    samples_m_scaled = samples_m / scaling

    # ellipse equation: a x^2 + b xy + c y^2 + d x + e y + f = 0
    er_params_scaled, er_center_scaled, er_angle_scaled, er_axis_scaled = compute_ellipse_params(samples_r_scaled)
    em_params_scaled, em_center_scaled, em_angle_scaled, em_axis_scaled = compute_ellipse_params(samples_m_scaled)

    er_params, er_center, er_angle, er_axis = compute_ellipse_params(samples_r)
    em_params, em_center, em_angle, em_axis = compute_ellipse_params(samples_m)

    sol_scaled = ellipse_intersection(er_params_scaled, em_params_scaled, Xs[0, 0:2]/scaling)
    sol = ellipse_intersection(er_params, em_params, sol_scaled*scaling)

    # find out intersection time of rocket with point
    T_intersect_rocket_index = np.argmin(np.linalg.norm(Xs[:,0:2] - sol, axis=1))
    T_intersect_rocket = Ts[T_intersect_rocket_index]

    # find out intersection time of rocket with point
    T_intersect_mars_index = np.argmin(np.linalg.norm(Xs[:,8:10] - sol, axis=1))
    T_intersect_mars = Ts[T_intersect_mars_index]

    if T_intersect_rocket > T_intersect_mars:
        # mars arrives at position before rocket
        # -> set initial position of mars back
        x0_planets[2] = - x0_planets[2]
        x0_planets[3] = - x0_planets[3]

        x0_planets[6] = - x0_planets[6]
        x0_planets[7] = - x0_planets[7]

        T_delta = T_intersect_rocket - T_intersect_mars
        sol = solve_ivp(fun= lambda t, x: odes.f_planets(t, x), t_span=(t0, T_delta), y0=x0_planets, method='Radau')
        X0_mars = sol.y[4:8, -1]
    else:
        # mars arrives after rocket
        # set initial position forward
        T_delta = T_intersect_mars - T_intersect_rocket
        # simulate from initial position for T_delta time
        sol = solve_ivp(fun= lambda t, x: odes.f_planets(t, x), t_span=(t0, T_delta), y0=x0_planets, method='Radau')
        X0_mars = sol.y[4:8, -1]

    #T_delta = T_intersect_mars - T_intersect_rocket
    #T_0_mars = T_intersect_mars - T_delta

    #if T_0_mars > 0:
        # we can just use the starting position from the existing simulation
    #    T_0_mars_index = np.argmin(np.abs(Ts - T_0_mars))
    #    X_0_mars = Xs[T_0_mars_index, 8:12]
    #else:
        # we have to simulate backward in time
    #    pass

    # find out intersection angle (= tangent of mars ellipse)
    angle_intersect_mars, normal_intersect_mars = intersection_angle(sol_scaled, em_params_scaled)
    angle_intersect_rocket, normal_intersect_rocket= intersection_angle(sol_scaled, er_params_scaled)

    # M = np.zeros((6,6))
    # for i in range(0,5):
    #     s = samples[i]
    #     M[i, 0] = s[0]**2
    #     M[i, 1] = s[1]**2
    #     M[i, 2] = s[0]*s[1]
    #     M[i, 3] = s[0]
    #     M[i, 4] = s[1]
    #     M[i, 5] = 1
    # M[-1, 5] = 1
    # b = np.zeros((6,1))
    # b[-1] = 1
    # #x = np.linalg.solve(M, np.zeros((5,1)))
    # params = np.linalg.lstsq(M, b)
    # params = params/min(params)
    #
    # pass



    # i = 0
    # for t in Ts:
    #     if pd.sun_dist(Xs[i, 0:2]) >= pd.mars_trajectory_radius:
    #         print("t = {}".format(t))
    #         pos = Xs[i, 0:2]
    #         print("x = {}".format(pos))
    #         print("angle = {}".format(np.arcsin(pos[1]/np.sqrt(pos[0]**2+pos[1]**2))))
    #         break
    #     i += 1

    #fa = FlightAnim(Ts, Xs)
    #fa.animate()



    vel_rocket = Xs[T_intersect_rocket_index, 2:4]
    vel_mars = Xs[T_intersect_mars_index, 10:12]
    delta_v = vel_mars - vel_rocket
    delta_v_norm = delta_v/np.linalg.norm(delta_v)
    e1 = np.array([1.0, 0.0])
    w_des = np.arccos(np.dot(delta_v_norm, e1))
    if delta_v_norm[1] < 0.0:
        # if vector points down in the y-component we have to adjust the angle such that it is mathematically positive
        w_des = 2.0 * np.pi - w_des
    #u_des = np.linalg.norm()
    a_des = delta_v/1200.0
    u_des = a_des[0]/np.cos(w_des)

    fp.ax.plot([sol[0], sol[0]+vel_mars[0]*1e8], [sol[1], sol[1]+vel_mars[1]*1e8])
    fp.ax.plot([sol[0], sol[0]+vel_rocket[0]*1e8], [sol[1], sol[1]+vel_rocket[1]*1e8])
    fp.ax.plot([sol[0], sol[0]+delta_v[0]*1e8], [sol[1], sol[1]+delta_v[1]*1e8])

    # tangent_intersect_mars = np.array([-normal_intersect_mars[1], normal_intersect_mars[0]])
    # tangent_intersect_rocket = np.array([-normal_intersect_rocket[1], normal_intersect_rocket[0]])
    # t_mars_scaled = tangent_intersect_mars*scaling
    # t_rocket_scaled = tangent_intersect_rocket*scaling
    # t_target = 2.0 * t_mars_scaled - t_rocket_scaled
    # fp.ax.plot([sol[0], sol[0]+t_mars_scaled[0]], [sol[1], sol[1]+t_mars_scaled[1]])
    # fp.ax.plot([sol[0], sol[0]+t_rocket_scaled[0]], [sol[1], sol[1]+t_rocket_scaled[1]])
    # fp.ax.plot([sol[0], sol[0]+t_target[0]], [sol[1], sol[1]+t_target[1]])

    em = Ellipse(em_center, em_axis[0], em_axis[1], em_angle, color='red', linestyle='--')
    er = Ellipse(er_center, er_axis[0], er_axis[1], er_angle, color='black')
    fp.ax.add_artist(em)
    fp.ax.add_artist(er)

   pass


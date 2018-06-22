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


def fitEllipse(x,y):
    # a x^2 + b xy + c y^2 + d x + e y + f = 0
    x = x[:,np.newaxis]
    y = y[:,np.newaxis]
    D =  np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
    S = np.dot(D.T,D)
    C = np.zeros([6,6])
    C[0,2] = C[2,0] = 2; C[1,1] = -1
    E, V =  eig(np.dot(inv(S), C))
    n = np.argmax(np.abs(E))
    a = V[:,n]
    return a

def ellipse_center(a):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    num = b*b-a*c
    x0=(c*d-b*f)/num
    y0=(a*f-b*d)/num
    return np.array([x0,y0])

def ellipse_angle_of_rotation( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    return 0.5*np.arctan(2*b/(a-c))

def ellipse_axis_length( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
    down1=(b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    down2=(b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    res1=np.sqrt(up/down1)
    res2=np.sqrt(up/down2)
    return np.array([res1, res2])

def compute_ellipse_params( samples ):
    params = fitEllipse(samples[:,0], samples[:,1])
    center = ellipse_center(params)
    angle = ellipse_angle_of_rotation(params)
    axis_length = 2.0 * ellipse_axis_length(params)

    return params, center, angle, axis_length

def ellipse_intersection( el1, el2, xguess=None):
    if xguess is None:
        xguess = np.zeros((2,1))
    sol = fsolve(ellipses, xguess, (el1, el2), fprime=d_ellipses, full_output=True, xtol=1.0e-11)
    print("solution = {}".format(sol[0]))
    print("value = {}".format(ellipses(sol[0], el1, el2)))

    return sol[0]

def ellipses(x, el1, el2):
    a1 = el1[0]
    b1 = el1[1]
    c1 = el1[2]
    d1 = el1[3]
    e1 = el1[4]
    f1 = el1[5]

    a2 = el2[0]
    b2 = el2[1]
    c2 = el2[2]
    d2 = el2[3]
    e2 = el2[4]
    f2 = el2[5]

    eq1 = a1 * x[0]**2 + b1 * x[0] * x[1] + c1 * x[1]**2 + d1 * x[0] + e1 * x[1] + f1
    eq2 = a2 * x[0]**2 + b2 * x[0] * x[1] + c2 * x[1]**2 + d2 * x[0] + e2 * x[1] + f2

    return np.array([eq1, eq2])

def d_ellipses(x, el1, el2):
    a1 = el1[0]
    b1 = el1[1]
    c1 = el1[2]
    d1 = el1[3]
    e1 = el1[4]
    f1 = el1[5]

    a2 = el2[0]
    b2 = el2[1]
    c2 = el2[2]
    d2 = el2[3]
    e2 = el2[4]
    f2 = el2[5]

    m00 = 2 * a1 * x[0] + b1 * x[1] + d1
    m01 = b1 * x[0] + 2 * c1 * x[1] + e1
    m10 = 2 * a2 * x[0] + b2 * x[1] + d2
    m11 = b2 * x[0] + 2 * c2 * x[1] + e2

    return np.array([[m00, m01], [m10, m11]])

def intersection_angle(point, ellipse_params):
    a = ellipse_params[0]
    b = ellipse_params[1]
    c = ellipse_params[2]
    d = ellipse_params[3]
    e = ellipse_params[4]
    f = ellipse_params[5]

    x = point[0]
    y = point[1]
    tangent_vector = np.array([2.0 * a * x + b * y + d, b * x + 2.0 * c * y + e])

    angle = 0.0
    return angle, tangent_vector

if __name__ == "__main__":

    # command array (time, angle, thrust)

    t0 = 0.0
    tf = 40000000.0
    dt = 600.0

    simulator = simulator.Simulator(pd)
    # commands = (time, duration, angle, thrust)
    commands = np.array([[0.0, 1200.0, 0.0, 5.0],[13352221.0, 1200.0, 4.4057, 3.8938]])
    #commands = np.array([[110.0, 1200.0, 0.0, 5.0],[1500.0, 100.0, 4.4057, 3.8938]])

    Ts, Xs = simulator.run_simulation(commands, tf)

    #fp = FlightPlot(Ts, Xs)
    #fp.plot()

    fa = FlightAnim(Ts, Xs)
    fa.animate()

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


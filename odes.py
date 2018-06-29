
import numpy as np
import time

from data import PlanetData

pd = PlanetData('GEO')
t_timer_start = 0.0
# ode for full simulation of rocket and all the planets
# state x contains
# x[0] position on x axis
# x[1] position on y axis
# x[2] velocity in x direction
# x[3] velocity in y direction
# etc. for planets
def f_full(t, x, u, w):

    p_r = np.array([x[0], x[1]]) # position of the rocket
    v_r = np.array([x[2], x[3]]) # velocity of the rocket

    p_E = np.array([x[4], x[5]]) # position of earth
    p_M = np.array([x[8], x[9]]) # position of mars
    p_S = np.array([x[12], x[13]]) # position of sun

    # calculate acceleration of rocket due to gravity
    a_r = pd.G * pd.m_S * (p_S - p_r)/np.linalg.norm(p_S - p_r)**3      # sun
    a_r += pd.G * pd.m_E * (p_E - p_r)/np.linalg.norm(p_E - p_r)**3    # earth
    a_r += pd.G * pd.m_M * (p_M - p_r)/np.linalg.norm(p_M - p_r)**3    # mars

    dp_r = v_r
    dv_r = a_r + u * np.array([np.cos(w), np.sin(w)])

    dx = np.zeros(x.shape)

    # rocket
    dx[0] = dp_r[0]
    dx[1] = dp_r[1]
    dx[2] = dv_r[0]
    dx[3] = dv_r[1]

    dx[4:16] = f_planets(t, x[4:16])

    return dx

def event_timeout(t, x):
    t_diff = time.clock() - t_timer_start
    t_max = 1.0
    if t_diff > t_max:
        print("Solver Timeout!")
        return 0.0
    else:
        return t_max - t_diff

def event_mars_hit(t, x):
    dist = 5.0e7
    vel_dist = 1.0e4

    dist_rocket_mars = np.linalg.norm(x[0:2]-x[8:10])
    vel_dist_rocket_mars = np.linalg.norm(x[2:4]-x[10:12])

    if dist_rocket_mars < dist  and vel_dist_rocket_mars < vel_dist:
        #print("Mars orbit reached!")
        return 0.0
    else:
        return dist_rocket_mars - dist + vel_dist_rocket_mars - vel_dist

def f_planets(t, x ):
    p_E = np.array([x[0], x[1]]) # position of earth
    v_E = np.array([x[2], x[3]]) # velocity of earth

    p_M = np.array([x[4], x[5]]) # position of mars
    v_M = np.array([x[6], x[7]]) # velocity of mars

    p_S = np.array([x[8], x[9]]) # position of sun
    v_S = np.array([x[10], x[11]]) # velocity of sun

    # acceleration of earth due to gravity
    a_E = pd.G * pd.m_S * (p_S - p_E)/np.linalg.norm(p_S - p_E)**3    # sun
    a_E += pd.G * pd.m_M * (p_M - p_E)/np.linalg.norm(p_M - p_E)**3   # mars
    dp_E = v_E
    dv_E = a_E

    # acceleration of mars due to gravity
    a_M = pd.G * pd.m_S * (p_S - p_M)/np.linalg.norm(p_S - p_M)**3    # sun
    a_M += pd.G * pd.m_E * (p_E - p_M)/np.linalg.norm(p_M - p_E)**3   # earth
    dp_M = v_M
    dv_M = a_M

    # acceleration of sun due to gravity
    a_S = pd.G * pd.m_E * (p_E - p_S)/np.linalg.norm(p_S - p_E)**3    # earth
    a_S += pd.G * pd.m_M * (p_M - p_S)/np.linalg.norm(p_M - p_S)**3   # mars
    dp_S = v_S
    dv_S = a_S

    dx = np.zeros(x.shape)

    # earth
    dx[0] = dp_E[0]
    dx[1] = dp_E[1]
    dx[2] = dv_E[0]
    dx[3] = dv_E[1]

    # mars
    dx[4] = dp_M[0]
    dx[5] = dp_M[1]
    dx[6] = dv_M[0]
    dx[7] = dv_M[1]

    # sun
    dx[8] = dp_S[0]
    dx[9] = dp_S[1]
    dx[10] = dv_S[0]
    dx[11] = dv_S[1]

    return dx

import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt

# ode for the rocket
# state x contains
# x[0] position on x axis
# x[1] position on y axis
# x[2] velocity in x direction
# x[3] velocity in y direction
def f(t, x):
    p = np.array([x[0], x[1]]) # position of the rocket
    v = np.array([x[2], x[3]]) # velocity of the rocket

    AE = 149597870700

    # position of sun and planets
    p_S = np.array([0.0, 0.0]) * AE # sun
    p_E = np.array([1.0, 0.0]) * AE # earth
    p_M = np.array([0.0, 1.5]) * AE # mars

    # mass of sun and planets
    m_S = 1.99e30 # sun
    m_E = 5.97e24 # earth
    m_M = 6.39e23 # mars

    G = 6.67408e-11

    # calculate acceleration due to gravity
    a = G * m_S * (p_S - p)/np.linalg.norm(p_S - p)**3      # sun
    #a += G * m_E * (p_E - p)/np.linalg.norm(p_E - p)**3    # earth
    #a += G * m_M * (p_M - p)/np.linalg.norm(p_M - p)**3    # mars

    dp = v
    dv = a + u * w

    dx = np.zeros(x.shape)
    dx[0] = dp[0]
    dx[1] = dp[1]
    dx[2] = dv[0]
    dx[3] = dv[1]

    return dx

if __name__ == "__main__":
    t0 = 0.0
    dt = 100000.0
    t1 = 0.5*365*24*60*60.0
    AE = 149597870700

    p0 = np.array([0.5, 0.5]) * 149597870700
    v0 = np.array([-20000.0, 20000.0])

    x0 = np.concatenate((p0, v0))

    integrator = ode(f)#.set_integrator('zvode')
    integrator.set_initial_value(x0, t0)

    n = int(np.ceil((t1-t0)/dt))+1
    Ts = np.zeros((n,1))
    Xs = np.zeros((n, len(x0)))
    Ts[0] = t0
    Xs[0] = x0
    i = 1
    while integrator.successful() and integrator.t < t1:
        Ts[i] = integrator.t + dt
        Xs[i] = integrator.integrate(integrator.t+dt)
        print(Ts[i], Xs[i])
        i += 1

    plt.axis('equal')
    plt.ylim([-AE, AE])
    plt.xlim([-AE, AE])
    plt.grid('on')
    plt.plot(Xs[:,0], Xs[:,1])
    plt.show()
    pass


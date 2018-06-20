import math
import numpy as np
from scipy.integrate import ode
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import time


# simulate circulation of earth / mars

def orb_E(t):
    AE = 149597870700.0      # starting at (AE,0)   circulation period 365 days
    
    
    p_E = np.array([ AE * math.cos(t*2*math.pi/31536000.0), 0.9999 * AE * math.sin(t*2*math.pi/31536000.0)])    

    return p_E
    
    
def orb_M(t):
    AE = 149597870700.0     # starting at (0,1.5173*AE)   circulation period 687 days
    
    
    p_M = np.array([ 1.524 * AE * math.sin(t*2.0*math.pi/59356800.0), 1.5173 * AE * math.cos(t*2.0*math.pi/59356800)])    

    return p_M


# ode for the rocket
# state x contains
# x[0] position on x axis
# x[1] position on y axis
# x[2] velocity in x direction
# x[3] velocity in y direction
def f(t, x, u, w): 

    p = np.array([x[0], x[1]]) # position of the rocket
    v = np.array([x[2], x[3]]) # velocity of the rocket

    AE = 149597870700

    # position of sun and planets
    p_S = np.array([0.0, 0.0]) * AE # sun
    # p_E = np.array([1.0, 0.0]) * AE # earth
    # p_M = np.array([0.0, 1.5173]) * AE # mars

    # mass of sun and planets
    m_S = 1.99e30 # sun
    m_E = 5.97e24 # earth
    m_M = 6.39e23 # mars

    G = 6.67408e-11

    # calculate acceleration due to gravity
    a = G * m_S * (p_S - p)/np.linalg.norm(p_S - p)**3      # sun
    a += G * m_E * (orb_E(t) - p)/np.linalg.norm(orb_E(t) - p)**3    # earth
    a += G * m_M * (orb_M(t) - p)/np.linalg.norm(orb_M(t) - p)**3    # mars
          
    
    dp = v
    dv = a + u * np.array([math.sin(w), math.cos(w)])

    dx = np.zeros(x.shape)
    dx[0] = dp[0]
    dx[1] = dp[1]
    dx[2] = dv[0]
    dx[3] = dv[1]

    return dx



if __name__ == "__main__":
    t0 = 0.0
    dt = 100000.0
    t1 = 2*365*24*60*60.0
    AE = 149597870700

    p0 = np.array([6770000.0 + AE, 0]) # rocket starts on a satellite orbit 400 km above sea level
    v0 = np.array([0, 7860.0])   # about first cosmic velocity
    

    x0 = np.concatenate((p0, v0))

    integrator = ode(f)#.set_integrator('zvode')
    integrator.set_initial_value(x0, t0)        

    n = int(np.ceil((2*365*24*60*60.0)/100000.0))
    u = np.zeros((n,1))    # thrust u 
    w = np.zeros((n,1))    # angle w of thrust u
    

    n = int(np.ceil((t1-t0)/dt))+1
    Ts = np.zeros((n,1))
    Xs = np.zeros((n, len(x0)))
    Ts[0] = t0
    Xs[0] = x0
    i = 1
    while integrator.successful() and integrator.t < t1:
        integrator.set_f_params(u[i-1],w[i-1])
        Ts[i] = integrator.t + dt        
        Xs[i] = integrator.integrate(integrator.t+dt)
       # print(Xs[i][2])
       # print(Ts[i], Xs[i])
        i += 1

    earth_orbit = Ellipse((0,0), 2*AE, 2*0.9999*AE)
    earth = Ellipse((1*AE,0), 0.1*AE, 0.1*AE)
    mars_orbit = Ellipse((0,0), 2*1.524*AE, 2*1.5173*AE)
    mars = Ellipse((0,1.5173*AE), 0.1*AE, 0.1*AE)
    sun = Ellipse((0,0),0.2*AE,0.2*AE)

    a = plt.subplot(111, aspect='equal')

    a.add_artist(mars_orbit)
    a.add_artist(earth_orbit)
    a.add_artist(sun)
    a.add_artist(earth)
    a.add_artist(mars)
    #earth_orbit.set_facecolor('blue')
    #earth_orbit.set_alpha(0.2)
    earth_orbit.set_linewidth(2)
    earth_orbit.set_fill(False)
    #mars_orbit.set_facecolor('red')
    #mars_orbit.set_alpha(0.2)
    mars_orbit.set_linewidth(2)
    mars_orbit.set_fill(False)
    sun.set_alpha(1)
    sun.set_facecolor('yellow')
    earth.set_alpha(1)
    earth.set_facecolor('blue') 
    mars.set_alpha(1)
    mars.set_facecolor('red')


#    fig = plt.figure()
#    ax1 = fig.add_subplot(1,1,1)

#    def animate(i):
#        xar = []
#        yar = []
#        for eachLine in Xs:
#            xar.append(Xs[i,0])
#            yar.append(Xs[i,1])
#        ax1.clear()
#        ax1.plot(xar,yar)

#    ani = anim.FuncAnimation(fig, animate, interval=1000)


    plt.axis('equal')
    plt.ylim([-2.5*AE, 2.5*AE])
    plt.xlim([-2.5*AE, 2.5*AE])
    plt.grid('on')
    plt.plot(Xs[:,0], Xs[:,1])
    plt.show()
    pass


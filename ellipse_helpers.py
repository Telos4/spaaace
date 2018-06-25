from scipy.optimize import fsolve
import numpy as np
from numpy.linalg import eig, inv


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
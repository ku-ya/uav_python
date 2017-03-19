#!/usr/bin/env python
import numpy as np
from scipy.integrate import odeint
import numpy.linalg as la
import pdb
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import sys
import seaborn as sns

class UAV(object):
  def __init__(self, J, e3):
    # self.J
    self.m = 4.34
    self.g = 9.81
    self.J = J
    self.e3 = e3
    self.kR = 8.81; # attitude gains
    self.kW = 2.54; # attitude gains
    self.kx = 16.*self.m; # position gains
    self.kv = 5.6*self.m;# position gains
    print('initialized')
  def dydt(self, X, t):
    R = np.reshape(X[0:9],(3,3));  # rotation from body to inertial
    W = X[9:12];   # angular rate
    x = X[12:15];  # position
    v = X[15:];    # velocity
    (f, M) = self.position_control(t, R, W, x, v);
    R_dot = np.dot(R,hat(W))
    W_dot = np.dot(la.inv(self.J), M - np.cross(W, np.dot(self.J, W)))
    x_dot = v
    v_dot = self.g*self.e3 - f*R.dot(self.e3)/self.m
    X_dot = np.concatenate((R_dot.flatten(), W_dot, x_dot, v_dot))
    return X_dot

  def position_control(self, t, R, W, x, v):
    (xd, xd_dot, xd_ddot, xd_dddot, xd_ddddot, b1d, b1d_dot, b1d_ddot) = desired_position(t);
    (ex, ev) = position_errors( x, xd, v, xd_dot)
    (Rd, Wd, Wd_dot) = self.desired_attitude(t, ex, ev, xd_ddot, xd_dddot, xd_ddddot, b1d, b1d_dot, b1d_ddot, R, W)
    (eR, eW) = attitude_errors( R, Rd, W, Wd )
    W_hat = hat(W)
    M= -self.kR*eR - self.kW*eW + np.cross(W, self.J.dot(W)) - self.J.dot(W_hat.dot(R.T.dot(Rd.dot(Wd))) - R.T.dot(Rd.dot(Wd_dot)))
    f = np.dot(self.kx*ex + self.kv*ev + self.m*self.g*self.e3 - self.m*xd_ddot, R.dot(self.e3) )
    return (f, M)

  def desired_attitude(self, t, ex, ev, xd_2dot, xd_3dot, xd_4dot, b1d, b1d_dot, b1d_ddot, R, W):
    e3 = self.e3
    kx = self.kx
    kv = self.kv
    m = self.m
    g = self.g
    hatW = hat(W)
    R_dot = R.dot(hatW)

    f = np.dot((kx*ex + kv*ev + m*g*e3 - m*xd_2dot),(np.dot(R, e3)))
    x_2dot = g*e3 - f*R.dot(e3)/m
    ex_2dot = x_2dot - xd_2dot

    f_dot = ( kx*ev + kv*ex_2dot - m*xd_3dot).dot(R.dot(e3)) + ( kx*ex + kv*ev + m*g*e3 - m*xd_2dot).dot(np.dot(R_dot,e3))
    x_3dot = -1/m*( f_dot*R + f*R_dot ).dot(e3)
    ex_3dot = x_3dot - xd_3dot

    A = -kx*ex - kv*ev - m*g*e3 + m*xd_2dot
    A_dot = -kx*ev - kv*ex_2dot + m*xd_3dot
    A_2dot = -kx*ex_2dot - kv*ex_3dot + m*xd_4dot

    norm_A = la.norm(A)
    b3c = - A/norm_A
    b3c_dot = - A_dot/norm_A + ( np.dot(A, A_dot)*A )/norm_A**3
    b3c_2dot = - A_2dot/norm_A + ( 2*np.dot(A*A_dot,A_dot) )/norm_A**3 + np.dot( A_dot* A_dot + A*A_2dot ,A)/norm_A**3 - 3*np.dot((A*A_dot)**2,A)/norm_A**5

    b_ = np.cross(b3c, b1d)
    b_dot = np.cross(b3c_dot, b1d) + np.cross(b3c, b1d_dot)
    b_2dot = np.cross(b3c_2dot, b1d) + 2*np.cross(b3c_dot, b1d_dot) + np.cross(b3c, b1d_ddot)

    b1c = -np.cross( b3c, np.cross(b3c, b1d) )/la.norm( np.cross(b3c, b1d) )
    b1c_dot = -( np.cross(b3c_dot, b_) + np.cross(b3c, b_dot) )/la.norm(b_) + np.cross(b3c, b_)*(b_dot* b_)/la.norm(b_)**3

    # intermediate steps to calculate b1c_2dot
    m_1 = ( np.cross(b3c_2dot, b_) + 2*np.cross(b3c_dot, b_dot) + np.cross(b3c, b_2dot) )/la.norm(b_)
    m_2 = ( np.cross(b3c_dot, b_) + np.cross(b3c, b_dot) )*np.dot(b_dot, b_)/la.norm(b_)**3
    m_dot = m_1 - m_2
    n_1 = np.cross(b3c, b_)*np.dot(b_dot, b_)
    n_1dot = ( np.cross(b3c_dot, b_) + np.cross(b3c, b_dot) )*np.dot(b_dot, b_) + np.cross(b3c, b_)*( np.dot(b_2dot, b_)+np.dot(b_dot, b_dot) )
    n_dot = n_1dot/la.norm(b_)**3 - 3*n_1*np.dot(b_dot, b_)/la.norm(b_)**5
    b1c_2dot = -m_dot + n_dot

    Rc = np.reshape([b1c, np.cross(b3c, b1c), b3c],(3,3))
    Rc_dot = np.reshape([b1c_dot, ( np.cross(b3c_dot, b1c) + np.cross(b3c, b1c_dot) ), b3c_dot],(3,3))
    Rc_2dot = np.reshape( [b1c_2dot, ( np.cross(b3c_2dot, b1c) + np.cross(b3c_dot, b1c_dot) + np.cross(b3c_dot, b1c_dot) + np.cross(b3c, b1c_2dot) ), b3c_2dot],(3,3))
    Wc = vee(Rc.T.dot(Rc_dot))
    Wc_dot= vee( Rc_dot.T.dot(Rc_dot) + Rc.T.dot(Rc_2dot))
    return (Rc, Wc, Wc_dot)

def vee(M):
  return np.array([M[2,1], M[0,2], M[1,0]])

def desired_position(t):
  xd = np.array([0, 0, 0])
  xd_dot = np.array([0, 0, 0])
  xd_ddot = np.array([0, 0, 0])
  xd_dddot = np.array([0, 0, 0])
  xd_ddddot = np.array([0, 0, 0])
  b1d = np.array([1, 0, 0])
  b1d_dot=np.array([0, 0, 0])
  b1d_ddot=np.array([0, 0, 0])
  return (xd, xd_dot, xd_ddot, xd_dddot, xd_ddddot, b1d, b1d_dot, b1d_ddot)

def attitude_errors( R, Rd, W, Wd ):
  eR = 0.5*vee(Rd.T.dot(R) - R.T.dot(Rd))
  eW = W - R.T.dot(Rd.dot(Wd))
  return (eR, eW)

def position_errors(x, xd, v, vd):
  ex = x - xd
  ev = v - vd
  return (ex, ev)

def hat(x):
  hat_x = [0, -x[2], x[1],
          x[2], 0, -x[0],
          -x[1], x[0], 0]
  return np.reshape(hat_x,(3,3))


if __name__ == "__main__":
  # execute only if run as a script
  J = np.diag([0.0820, 0.0845, 0.1377])
  e3 = np.array([0.,0.,1.])
  uav_t = UAV(J, e3)
  t_max = 4
  N = 100*t_max + 1
  t = np.linspace(0,t_max,N)
  xd = np.array([0.,0.,0.])
  # Initial Conditions
  R0 = [[1., 0., 0.],
    [0., -0.9995, -0.0314],
    [0., 0.0314, -0.9995]] # initial rotation
  W0 = [0.,0.,0.];   # initial angular velocity
  x0 = [0.,0.,0.];  # initial position (altitude?0)
  v0 = [0.,0.,0.];   # initial velocity
  R0v = np.array(R0).flatten().T
  y0 = np.concatenate((R0v, W0,x0,v0))
  sim = odeint(uav_t.dydt,y0,t)
  # fig, ax = plt.subplots()
  fig = plt.figure()
  ax = p3.Axes3D(fig)

  # x = np.arange(0, 2*np.pi, 0.01)
  line, = ax.plot(sim[:,-6], sim[:,-5])
  def animate(i):
    line.set_data(np.vstack([sim[:i,-6], sim[:i,-5]]))  # update the data
    line.set_3d_properties(sim[:i,-4])
    return line,

  # Setting the axes properties
  ax.set_xlim3d([-1.0, 1.0])
  ax.set_xlabel('X')
  ax.set_ylim3d([-1.0, 1.0])
  ax.set_ylabel('Y')
  ax.set_zlim3d([0.0, 1.0])
  ax.set_zlabel('Z')
  # Init only required for blitting to give a clean slate.
  # def init():
    # print(np.concatenate((x.T,np.ma.array(x, mask=True).T)).shape)
    # pdb.set_trace()
    # line.set_data(np.concatenate((x,np.ma.array(x, mask=True))))
    # line.set_3d_properties(x)
    # return line,
  ani = animation.FuncAnimation(fig, animate, np.arange(N),
                              interval=25, blit=False)
  # plt.plot(t,sim[:,-6:-3])
  plt.grid()
  plt.show()
  # x = xd
  # uav_t.position_control(xd,x)
  # ode()

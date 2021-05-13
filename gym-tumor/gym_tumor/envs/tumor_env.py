import gym
from gym import error, spaces, utils
from gym.utils import seeding
import math
import numpy as np

class TumorEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):


    self.a1 = 0.2
    self.a2 = 0.3
    self.a3 = 0.1
    self.b1 = 1.0
    self.b2 = 1.0
    self.alpha = 0.3
    self.c1 = 1.0
    self.c2 = 0.5
    self.c3 = 1.0
    self.c4 = 1.0
    self.d1 = 0.2
    self.d2 = 1.0
    self.r1 = 1.5
    self.r2 = 1.0
    self.s = 0.33
    self.ro = 0.01




    self.x0dot = 1.0
    self.x1dot = 0.7
    self.x2dot = 1.0
    self.x3dot = 0.


    self.T = 100
    self.xd = 0

    self.x0_r = 4.5#0.56
    self.x1_r = 1.0 #0.44
    self.x2_r = 1.0#0.44
    self.x3_r = 0.

    self.h = 10.  # ������������ �������� ����������

    self.max_u = 10.
    self.min_u = 0.
    self.dt = 0.1



    self.low_obs = np.array([0., 0., 0., 0.], dtype=np.float32)
    self.high_obs = np.array([self.h, self.h, self.h, self.h], dtype=np.float32)


    low_act = np.array([self.min_u], dtype=np.float32)
    high_act = np.array([self.max_u], dtype=np.float32)

    self.action_space = spaces.Box(
          low=low_act,
          high=high_act,
          dtype=np.float32
      )

    self.observation_space = spaces.Box(
          low=self.low_obs,
          high=self.high_obs,
          dtype=np.float32
      )

    self.seed()

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def step(self, u):

    x0, x1, x2, x3 = self.state
    x0dot = self.x0dot
    x1dot = self.x1dot
    x2dot = self.x2dot
    x3dot = self.x3dot
    #y3dot = 0


    a1 = self.a1
    a2 = self.a2
    a3 = self.a3
    b1 = self.b1
    b2 = self.b2
    alpha = self.alpha
    c1 = self.c1
    c2 = self.c2
    c3 = self.c3
    c4 = self.c4
    d1 = self.d1
    d2 = self.d2
    r1 = self.r1
    r2 = self.r2
    s = self.s
    ro = self.ro


    dt = self.dt


    u = np.clip(u, self.min_u, self.max_u)[0]


    self.last_u = u  # for rendering

    new_x0dot = x0dot + (r1*x0*(1 - b1*x0) - c2*x2*x1 - c3*x0*x1 - a2*x0*x3) * dt #(1-np.exp(-x3))
    new_x1dot = x1dot + (r2*x1*(1 - b2*x1) - c4*x0*x1 - a3*x3*x1)*dt
    new_x2dot = x2dot + (s + ro*x2*x0/(alpha + x0) - c1*x2*x0 - d1*x2 - a1*x3*x2)*dt
    #new_x0dot = x0dot + (a*x0*(1 - b*x0) - c1*x1*x0 - Kt*x3*x0) * dt
    #new_x1dot = x1dot + (alpha1 - f*x1 + g * x0 / (h+x0)*x1 - p*x1*x0 - Kn*x1*x3)*dt
    #new_x2dot = x2dot + (alpha2 - betta*x2 - Kc*x3*x2)*dt
    new_x3dot = x3dot + (-d2*x3 + u)*dt

    #new_y3dot = y3dot + (y0 - (r1/2*u1**2 + r2/2*u2**2))*dt

    new_x0 = x0 + new_x0dot * dt
    new_x1 = x1 + new_x1dot * dt
    new_x2 = x2 + new_x2dot * dt
    new_x3 = x3 + new_x3dot * dt
    #print('2', new_x0, new_x1, new_x2)

    self.x0dot = new_x0dot
    self.x1dot = new_x1dot
    self.x2dot = new_x2dot
    self.x3dot = new_x3dot


    #print(costs)

    #terminal = self._terminal()

    bol1 = x1 < 0.4
    bol2 = x2 < 0.4
    costs =  (-x0 - 0.8*bol1 - 0.8*bol2) #- 40*terminal C#np.log(x0/new_x0)-abs(x0-self.xd) - 200*terminal

    new_x0 = np.clip(new_x0, 0., self.h)
    new_x1 = np.clip(new_x1, 0., self.h)
    new_x2 = np.clip(new_x2, 0., self.h)
    new_x3 = np.clip(new_x3, 0., self.h)




    self.state = np.array([new_x0, new_x1, new_x2, new_x3])
    return self._get_obs(), costs, False, {}


  def render(self, mode='human'):
    return 0

  def _terminal(self):
      x0, x1, x2, x3 = self.state
      return bool(x1<0.4 or x2 < 0.4)

  def close(self):
    return 0

  def reset(self):
    #print(self.np_random.uniform(low=self.low_obs, high=self.high_obs))
    self.state = [self.np_random.uniform(low=0, high=self.h), self.x1_r, self.x2_r, self.x3_r] #self.np_random.uniform(low=self.low_obs, high=self.high_obs)

    self.x0dot = 0.
    self.x1dot = 0.
    self.x2dot = 0.
    self.x3dot = 0.

    self.last_u = None
    return self._get_obs()

  def _get_obs(self):
    x0, x1, x2, x3 = self.state
    return np.array([x0, x1, x2, x3])

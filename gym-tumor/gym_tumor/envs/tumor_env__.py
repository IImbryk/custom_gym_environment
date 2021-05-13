import gym
from gym import error, spaces, utils
from gym.utils import seeding
import math
import numpy as np

class TumorEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):

    self.a = 4.31*10**(-2)
    self.b = 1.02*10**(-14)
    self.c1 = 3.41*10**(-10)
    self.f = 4.12 * 10**(-2)
    self.g = 4.5 * 10**(-2)
    self.h = 2.02 * 10
    self.Kc = 6.00 * 10**(-1)
    self.Kn = 6.00 * 10**(-1)
    self.Kt = 8.00 * 10**(-1)
    self.p = 2.00 * 10**(-11)
    self.alpha1 = 1.2 * 10**4
    self.alpha2 = 7.50 * 10**8
    self.betta = 1.20 * 10**(-2)
    self.gamma = 9.00 * 10**(-1)

    self.d1 = 0.2


    self.x0dot = 0.
    self.x1dot = 0.
    self.x2dot = 0.
    self.x3dot = 0.


    self.T = 100
    self.xd = 0

    self.x0_r = 0.56
    self.x1_r = 0.44
    self.x2_r = 0.44
    self.x3_r = 0.

    self.h = 30.  # ������������ �������� ����������

    self.max_u = 5.
    self.min_u = 0.
    self.dt = .01



    self.low_obs = np.array([0.001, 0.001, 0.001, 0.], dtype=np.float32)
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


    a = self.a
    b = self.b
    c1 = self.c1
    f = self.f
    g = self.g
    h = self.h
    Kc = self.Kc
    Kn = self.Kn
    Kt = self.Kt
    p = self.p
    alpha1 = self.alpha1
    alpha2 = self.alpha2
    betta = self.betta
    gamma = self.gamma

    dt = self.dt


    u = np.clip(u, self.min_u, self.max_u)[0]


    self.last_u = u  # for rendering

    new_x0dot = x0dot + (a*x0*(1 - b*x0) - c1*x1*x0 - Kt*(1-math.exp(-x3))*x0) * dt
    new_x1dot = x1dot + (alpha1 - f*x1 + g * x0 / (h+x0)*x1 - p*x1*x0 - Kn*(1-math.exp(-x3))*x1)*dt
    new_x2dot = x2dot + (alpha2 - betta*x2 - Kc*(1-math.exp(-x3))*x2)*dt
    #new_x0dot = x0dot + (a*x0*(1 - b*x0) - c1*x1*x0 - Kt*x3*x0) * dt
    #new_x1dot = x1dot + (alpha1 - f*x1 + g * x0 / (h+x0)*x1 - p*x1*x0 - Kn*x1*x3)*dt
    #new_x2dot = x2dot + (alpha2 - betta*x2 - Kc*x3*x2)*dt
    new_x3dot = x3dot + (-gamma*x3 + u)*dt

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



    costs = np.log(x0/new_x0)

    new_x0 = np.clip(new_x0, 0.01, self.h)
    new_x1 = np.clip(new_x1, 0.01, self.h)
    new_x2 = np.clip(new_x2, 0.01, self.h)
    new_x3 = np.clip(new_x3, 0.01, self.h)

    #terminal = self._terminal()


    self.state = np.array([new_x0, new_x1, new_x2, new_x3])
    return self._get_obs(), costs, False, {}


  def render(self, mode='human'):
    return 0

  def _terminal(self):
      x0, x1, x2, x3 = self.state
      return bool(x0<800 or x2>120)

  def close(self):
    return 0

  def reset(self):

    self.state = [self.x0_r, self.x1_r, self.x2_r, self.x3_r] #self.np_random.uniform(low=self.low_obs, high=self.high_obs)

    self.x0dot = 0
    self.x1dot = 0
    self.x2dot = 0
    self.x3dot = 0

    self.last_u = None
    return self._get_obs()

  def _get_obs(self):
    x0, x1, x2, x3 = self.state
    return np.array([x0, x1, x2, x3])

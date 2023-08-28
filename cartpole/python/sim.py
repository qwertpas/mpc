import sys, os
root_dir = os.path.dirname(os.path.abspath(__file__))+'/../..'
sys.path.append(root_dir)

import numpy as np
import cartPoleEnv
from blitting import LivePlot



env = cartPoleEnv.CartPoleEnv()
env.gravity = 9.8
env.masscart = 1.0
env.masspole = 0.1
env.length = 0.5  # height of center of mass
env.max_force = 10.0
env.reset()


lp = LivePlot(
    labels=(
        ('x',), 
        ('theta',),
        ('xdot',), 
        ('thetadot',),
        ('u',)
    ),
    ymins=[-2.4, -env.theta_threshold_radians, -1, -1, -env.max_force],
    ymaxes=[2.4, env.theta_threshold_radians, 1, 1, env.max_force],
)

u = 0

running = True
while not lp.closed:
    obs, reward, terminated, state = env.step(u)
    if(terminated):
        env.reset()


    K = [0.5, 0.3, 20, 1]
    u = K@state

    x, xdot, theta, thetadot = state


    lp.plot(
        x, 
        theta,
        xdot,
        thetadot,
        u,
    )

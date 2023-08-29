import sys, os
root_dir = os.path.dirname(os.path.abspath(__file__))+'/../..'
sys.path.append(root_dir)

from blitting import LivePlot
import cartPoleEnv
from controllers import cartpole_LQR, cartpole_MPC
import numpy as np

##################   CONSTANTS   ####################
mass_cart = 1
mass_pole = 0.1
pole_length = 0.5
u_max = 10
#####################################################

env = cartPoleEnv.CartPoleEnv()
env.masscart = mass_cart
env.masspole = mass_pole
env.length = pole_length
env.max_force = u_max
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

lqr = cartpole_LQR(M=mass_cart, m=mass_pole, L=pole_length)
mpc = cartpole_MPC(M=mass_cart, m=mass_pole, L=pole_length, dt=0.1, N=20)

u = 0
running = True
while not lp.closed:
    obs, reward, terminated, state = env.step(u)
    if(terminated):
        u = 0
        env.reset()
        continue

    # u = lqr.get_u(state)
    u = mpc.solve_u(state)
    print (u)
    x, xdot, theta, thetadot = state



    lp.plot(
        x, 
        theta,
        xdot,
        thetadot,
        u,
    )

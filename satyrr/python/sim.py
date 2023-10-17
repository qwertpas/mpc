import numpy as np
from numpy import sin, cos, pi
import matplotlib.pyplot as plt
import imageio
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)

q = np.array([0, 0.05, 0.5, pi/6, 4*pi/6])  # wheel x, pendulum angle, leg length, shoulder angle, elbow angle
qd = np.array([0., 0., 0., 0., 0.])
dt = 0.01

m = np.array([2*0.324, 10.9, 2*1.96])  # wheel, body, hand masses
L = np.array([0.109, 0.200])  # upper arm, forearm lengths
N = np.array([6, 6, 6])  # wheel, shoulder, elbow gear ratios
I = np.array([2*1.62e-3, 2.30e-1, 0.00])  # wheel, body, hand structure inertias
J = np.array([2*6.07e-5, 6.07e-5, 2*6.07e-5])  # wheel, body, hand rotor inertias
g = 9.81
r = 0.09525

q_log = []
tau_log = []


# Simulation loop
for i in range(500):

    # Mass matrix of the manipulator equation: M*qdd + C + G = τ
    M = np.matrix([
        [
            J[0] * N[0] ** 2 + m[0] + m[1] + m[2],
            (cos(q[1] + q[3]) * L[0] + cos(q[1] + q[3] + q[4]) * L[1] - cos(q[1]) * q[2]) * m[2] - cos(q[1]) * m[1] * q[2],
            (-m[1] - m[2]) * sin(q[1]),
            (cos(q[1] + q[3]) * L[0] + cos(q[1] + q[3] + q[4]) * L[1]) * m[2],
            cos(q[1] + q[3] + q[4]) * L[1] * m[2],
        ],
        [
            (cos(q[1] + q[3]) * L[0] + cos(q[1] + q[3] + q[4]) * L[1] - cos(q[1]) * q[2]) * m[2] - cos(q[1]) * m[1] * q[2],
            ((sin(q[1] + q[3]) * L[0] + sin(q[1] + q[3] + q[4]) * L[1] - sin(q[1]) * q[2]) ** 2 + (cos(q[1] + q[3]) * L[0] + cos(q[1] + q[3] + q[4]) * L[1] - cos(q[1]) * q[2]) ** 2) * m[2] + I[0] + I[1] + I[2] + m[1] * q[2] ** 2,
            (sin(q[3] + q[4]) * L[1] + sin(q[3]) * L[0]) * m[2],
            -cos(q[3] + q[4]) * L[1] * m[2] * q[2] - cos(q[3]) * L[0] * m[2] * q[2] + 2 * cos(q[4]) * L[0] * L[1] * m[2] + I[1] + I[2] + L[0] ** 2 * m[2] + L[1] ** 2 * m[2],
            -cos(q[3] + q[4]) * L[1] * m[2] * q[2] + cos(q[4]) * L[0] * L[1] * m[2] + I[2] + L[1] ** 2 * m[2],
        ],
        [
            (-m[1] - m[2]) * sin(q[1]),
            (sin(q[3] + q[4]) * L[1] + sin(q[3]) * L[0]) * m[2],
            m[1] + m[2],
            (sin(q[3] + q[4]) * L[1] + sin(q[3]) * L[0]) * m[2],
            sin(q[3] + q[4]) * L[1] * m[2],
        ],
        [
            (cos(q[1] + q[3]) * L[0] + cos(q[1] + q[3] + q[4]) * L[1]) * m[2],
            -cos(q[3] + q[4]) * L[1] * m[2] * q[2] - cos(q[3]) * L[0] * m[2] * q[2] + 2 * cos(q[4]) * L[0] * L[1] * m[2] + I[1] + I[2] + L[0] ** 2 * m[2] + L[1] ** 2 * m[2],
            (sin(q[3] + q[4]) * L[1] + sin(q[3]) * L[0]) * m[2],
            ((sin(q[1] + q[3]) * L[0] + sin(q[1] + q[3] + q[4]) * L[1]) ** 2 + (cos(q[1] + q[3]) * L[0] + cos(q[1] + q[3] + q[4]) * L[1]) ** 2) * m[2] + I[1] + I[2] + J[1] * N[1] ** 2,
            cos(q[4]) * L[0] * L[1] * m[2] + I[2] + L[1] ** 2 * m[2],
        ],
        [ 
            cos(q[1] + q[3] + q[4]) * L[1] * m[2],
            -cos(q[3] + q[4]) * L[1] * m[2] * q[2] + cos(q[4]) * L[0] * L[1] * m[2] + I[2] + L[1] ** 2 * m[2], 
            sin(q[3] + q[4]) * L[1] * m[2],
            cos(q[4]) * L[0] * L[1] * m[2] + I[2] + L[1] ** 2 * m[2],
            I[2] + J[2] * N[2] ** 2 + L[1] ** 2 * m[2],
        ],
    ])
    M_inv = np.linalg.inv(M)

    # Sum of Coriolis and Gravity matrices
    CG = np.matrix([
        [
            (
                (sin(q[1]) * qd[1] * q[2] - cos(q[1]) * qd[2]) * m[1]
                - (
                    (sin(q[1] + q[3]) * L[0] + sin(q[1] + q[3] + q[4]) * L[1]) * qd[3]
                    + (
                        sin(q[1] + q[3]) * L[0]
                        + sin(q[1] + q[3] + q[4]) * L[1]
                        - sin(q[1]) * q[2]
                    )
                    * qd[1]
                    + sin(q[1] + q[3] + q[4]) * L[1] * qd[4]
                    + cos(q[1]) * qd[2]
                )
                * m[2]
            )
            * qd[1]
            - (m[1] + m[2]) * cos(q[1]) * qd[1] * qd[2]
            - (
                (sin(q[1] + q[3]) * L[0] + sin(q[1] + q[3] + q[4]) * L[1]) * qd[1]
                + (sin(q[1] + q[3]) * L[0] + sin(q[1] + q[3] + q[4]) * L[1]) * qd[3]
                + sin(q[1] + q[3] + q[4]) * L[1] * qd[4]
            )
            * qd[3]
            * m[2]
            - (qd[1] + qd[3] + qd[4]) * sin(q[1] + q[3] + q[4]) * L[1] * qd[4] * m[2]
        ],
        [
            g * sin(q[1] + q[3]) * L[0] * m[2]
            + g * sin(q[1] + q[3] + q[4]) * L[1] * m[2]
            - g * sin(q[1]) * m[1] * q[2]
            - g * sin(q[1]) * m[2] * q[2]
            + 2 * sin(q[3] + q[4]) * L[1] * qd[1] * qd[3] * m[2] * q[2]
            + 2 * sin(q[3] + q[4]) * L[1] * qd[1] * qd[4] * m[2] * q[2]
            + sin(q[3] + q[4]) * L[1] * qd[3] ** 2 * m[2] * q[2]
            + 2 * sin(q[3] + q[4]) * L[1] * qd[3] * qd[4] * m[2] * q[2]
            + sin(q[3] + q[4]) * L[1] * qd[4] ** 2 * m[2] * q[2]
            + 2 * sin(q[3]) * L[0] * qd[1] * qd[3] * m[2] * q[2]
            + sin(q[3]) * L[0] * qd[3] ** 2 * m[2] * q[2]
            - 2 * sin(q[4]) * L[0] * L[1] * qd[1] * qd[4] * m[2]
            - 2 * sin(q[4]) * L[0] * L[1] * qd[3] * qd[4] * m[2]
            - sin(q[4]) * L[0] * L[1] * qd[4] ** 2 * m[2]
            - 2 * cos(q[3] + q[4]) * L[1] * qd[1] * qd[2] * m[2]
            - 2 * cos(q[3]) * L[0] * qd[1] * qd[2] * m[2]
            + 2 * qd[1] * qd[2] * m[1] * q[2]
            + 2 * qd[1] * qd[2] * m[2] * q[2]
        ],
        [
            g * cos(q[1]) * m[1]
            + g * cos(q[1]) * m[2]
            + cos(q[3] + q[4]) * L[1] * qd[1] ** 2 * m[2]
            + 2 * cos(q[3] + q[4]) * L[1] * qd[1] * qd[3] * m[2]
            + 2 * cos(q[3] + q[4]) * L[1] * qd[1] * qd[4] * m[2]
            + cos(q[3] + q[4]) * L[1] * qd[3] ** 2 * m[2]
            + 2 * cos(q[3] + q[4]) * L[1] * qd[3] * qd[4] * m[2]
            + cos(q[3] + q[4]) * L[1] * qd[4] ** 2 * m[2]
            + cos(q[3]) * L[0] * qd[1] ** 2 * m[2]
            + 2 * cos(q[3]) * L[0] * qd[1] * qd[3] * m[2]
            + cos(q[3]) * L[0] * qd[3] ** 2 * m[2]
            - qd[1] ** 2 * m[1] * q[2]
            - qd[1] ** 2 * m[2] * q[2]
        ],
        [
            (
                g * sin(q[1] + q[3]) * L[0]
                + g * sin(q[1] + q[3] + q[4]) * L[1]
                - sin(q[3] + q[4]) * L[1] * qd[1] ** 2 * q[2]
                - sin(q[3]) * L[0] * qd[1] ** 2 * q[2]
                - 2 * sin(q[4]) * L[0] * L[1] * qd[1] * qd[4]
                - 2 * sin(q[4]) * L[0] * L[1] * qd[3] * qd[4]
                - sin(q[4]) * L[0] * L[1] * qd[4] ** 2
                - 2 * cos(q[3] + q[4]) * L[1] * qd[1] * qd[2]
                - 2 * cos(q[3]) * L[0] * qd[1] * qd[2]
            )
            * m[2]
        ],
        [
            (
                g * sin(q[1] + q[3] + q[4])
                - sin(q[3] + q[4]) * qd[1] ** 2 * q[2]
                + sin(q[4]) * L[0] * qd[1] ** 2
                + 2 * sin(q[4]) * L[0] * qd[1] * qd[3]
                + sin(q[4]) * L[0] * qd[3] ** 2
                - 2 * cos(q[3] + q[4]) * qd[1] * qd[2]
            )
            * L[1]
            * m[2]
        ],
    ])


    legforce = 500*(0.5 - q[2]) - 100*(qd[2]) + 150 #pd controller on leg length = 0.5m
    shoulderforce = -5*qd[3]                  #damping
    elbowforce = -5*qd[4]                     #damping
    linangle = 0

    if i < 150:
        elbowforce += 10*(pi/3 - q[4])
        shoulderforce += 10*(pi/3 - q[3])
        linangle = -0.11
    elif i > 150 and i < 210:                     #do a squat
        legforce = 500*(0.2 - q[2]) - 100*(qd[2]) + 150 #pd controller on leg length = 0.5m 
        elbowforce += 20*(pi/6 - q[4])
        shoulderforce += 20*(pi/3 - q[3])
        linangle = -0.2
    elif i > 210 and i < 220:
        elbowforce += 20*(pi/6 - q[4])
        shoulderforce += 20*(pi/3 - q[3])
        linangle = -0.5
    elif i > 230 and i < 240:
        shoulderforce += 20*(3*pi/4 - q[3])
        elbowforce += 20*(pi/2 - q[4])
        linangle = -0.5
    elif i > 240 and i < 300:
        shoulderforce += 20*(pi/2 - q[3])
        elbowforce += 20*(pi/2 - q[4])
        linangle = -0.2
    elif i > 300:
        shoulderforce += 20*(pi/2 - q[3])
        elbowforce += 20*(pi/2 - q[4])
        linangle = -0.1



    K = [10, 10, -300, -5]
    # K = [0, 0, 0, 0]
    cartstate = np.array([q[0], qd[0], q[1]+linangle, qd[1]]) #LQR(ish) on the cartpole (not optimized)
    cartforce = K @ cartstate

    tau = np.array([cartforce, 0, legforce, shoulderforce, elbowforce]).reshape((5,1))

    # Solve manipulator equation for qdd (acceleration)
    qdd_new = np.asarray(M_inv@(tau - CG)).flatten() 
    
    # Euler integration, switch to RK4 later for more accuracy
    qd += qdd_new * dt
    q += qd * dt

    #log states and joint torques
    q_log.append(q.tolist())
    tau_log.append(tau.tolist())

q_log = np.array(q_log)
tau_log = np.array(tau_log)


# Plot the states and torques over time
plot_log = False
plot_log = True
if plot_log:
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10,8))
    axs[0].plot(q_log[:,0], label='x')
    axs[0].plot(q_log[:,1], 'k', label='θ')
    axs[0].plot(q_log[:,2], label='leg')
    axs[0].plot(q_log[:,3], label='shoulder')
    axs[0].plot(q_log[:,4], label='elbow')

    axs[1].plot(tau_log[:,0], label='τ wheel')
    axs[1].plot(tau_log[:,2], label='τ leg')
    axs[1].plot(tau_log[:,3], label='τ shoulderforce')
    axs[1].plot(tau_log[:,4], label='τ elbow')

    for ax in axs:
        ax.legend()
    plt.show()

create_gif = False
create_gif = True
# Create GIF using logged states and forward kinematics
if create_gif:
    png_dir = f"{dir_path}/log"
    for i in range(0, len(q_log), 10):
        q = q_log[i]
        p_wheel = [q[0], 0]
        p_body = [-sin(q[1])*q[2] + q[0], cos(q[1])*q[2]]
        p_elbow = [sin(q[1] + q[3])*L[0] - sin(q[1])*q[2] + q[0], -cos(q[1] + q[3])*L[0] + cos(q[1])*q[2]]
        p_hand = [sin(q[1] + q[3])*L[0] + sin(q[1] + q[3] + q[4])*L[1] - sin(q[1])*q[2] + q[0], -cos(q[1] + q[3])*L[0] - cos(q[1] + q[3] + q[4])*L[1] + cos(q[1])*q[2]]

        fig, ax = plt.subplots()
        def plot_link(ax, a, b, c='k'):
            ax.plot([a[0], b[0]], [a[1], b[1]], c=c)

        plot_link(ax, (-1,-r), (1,-r))
        plot_link(ax, p_wheel, p_body, 'r')
        plot_link(ax, p_body, p_elbow, 'g')
        plot_link(ax, p_elbow, p_hand, 'b')

        ax.add_artist(plt.Circle((p_wheel[0], p_wheel[1]), r, color='gray'))
        ax.add_artist(plt.Circle((p_body[0], p_body[1]), 0.055, color='gray'))
        ax.add_artist(plt.Circle((p_hand[0], p_hand[1]), 0.015, color='black'))



        ax.set_title(f"t={round(i*0.01,2)}s")
        ax.set_xlim([-1, 1])
        ax.set_ylim([-0.5, 1.5])
        ax.set_aspect('equal')
        plt.savefig(f"{png_dir}/frame_{i:04d}.png")
        plt.close()

    images = []
    for file_name in sorted(os.listdir(png_dir)):
        if file_name.endswith('.png'):
            file_path = os.path.join(png_dir, file_name)
            images.append(imageio.imread(file_path))
    for _ in range(10): #add 10 of the last frame at the end
        images.append(imageio.imread(file_path))

    imageio.mimsave(f'{dir_path}/gif/sim.gif', images, loop=0)
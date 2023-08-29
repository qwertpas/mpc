from scipy.linalg import block_diag, solve_continuous_are, pinv, expm
import numpy as np
from numpy import array, eye, hstack, vstack, diag, zeros, reshape, full, Inf
np.set_printoptions(precision=2, suppress=True)
                       
g = 9.81

import qpsolvers
from qpsolvers import solve_qp


class cartpole_LQR():

    def __init__(self, M, m, L):
        '''
        Cart (M) with point-mass (m) on a pole (L)
        '''
        self.Q = diag([1, 1, 1, 1])  # weight for state error
        self.R = eye(1)              # weight for actuation

        # https://danielpiedrahita.wordpress.com/portfolio/cart-pole-control/
        # state vector X = [x, theta, xdot, thetadot]
        # linearized dynamics at X = [0, 0, 0, 0]
        self.A = array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, -m*g/M, 0, 0],
            [0, (m*g+M*g)/(M*L), 0, 0]
        ])
        self.B = array([
            [0],
            [0],
            [1/M],
            [1/(M*L)]
        ])

        self.P = solve_continuous_are(self.A, self.B, self.Q, self.R)

        # self.K = np.dot(np.linalg.inv(self.R), np.dot(self.B.T, self.P))
        self.K = array([0.5, 0.3, 20, 1]) #???

    def get_u(self, state):
        return self.K @ state
    


class cartpole_MPC():

    def __init__(self, M, m, L, dt, N):
        '''
        Cart (M) with point-mass (m) on a pole (L)
        '''


        self.N = N
        self.x_costs = array([1, 1, 1, 1])
        self.u_cost = 0.1
        self.u_min = -10
        self.u_max = 10


        self.A = array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, -m*g/M, 0, 0],
            [0, (m*g+M*g)/(M*L), 0, 0]
        ])
        self.B = array([
            [0],
            [0],
            [1/M],
            [1/(M*L)]
        ])

        # self.A = 9*np.random.random((4,4))
        # self.B = 9*np.random.random((4,1))
        
        self.A_dis = expm(self.A*dt)
        # self.B_dis = pinv(self.A) @ (self.A_dis - eye(4)) @ self.B
        self.B_dis = dt * self.B

        # self.X = zeros(4*N)

        # self.A_dis = self.A
        # self.B_dis = self.B


    def solve(self, state, state_des=zeros(4)):

        N = self.N
        x_costs = self.x_costs
        u_cost = self.u_cost
        A_dis = self.A_dis
        B_dis = self.B_dis
        u_min = self.u_min
        u_max = self.u_max

        state = reshape(state, (4, 1))
        state_des = reshape(state_des, (4, 1))

        Q = diag(x_costs)
        f = -2 * state_des.flatten() * x_costs
        for i in range(N): #Q becomes 4(N+1) x 4(N+1)
            Q = block_diag(Q, diag(x_costs))
            f = hstack((f, -2 * state_des.flatten() * x_costs))

        R = u_cost * eye(N)
        H = block_diag(Q, R)

        f_aug = hstack((f, zeros(N)))

        A_eq = hstack((eye(4), zeros((4,4*N)), zeros((4,N))))
        for i in range(N):
            eq_constraint = hstack((zeros((4,4*i)), -A_dis, eye(4), zeros((4,4*(N-1-i)+i)), -B_dis, zeros((4,(N-1-i)))))
            A_eq = vstack((A_eq, eq_constraint))

        b_eq = vstack((state, zeros((4*N, 1))))

        # print("A_eq: \n", A_eq, A_eq.shape, '\n')
        # print("b_eq: \n", b_eq, b_eq.shape, '\n')

        lower_bound = vstack((full((4*(N+1),1), -999), full((N,1), u_min)))
        upper_bound = vstack((full((4*(N+1),1), +999), full((N,1), u_max)))

        # print(qpsolvers.available_solvers)

        self.X = solve_qp(H, f_aug, A=A_eq, b=b_eq, lb=lower_bound, ub=upper_bound, solver='daqp')
        return self.X
    
    def solve_u(self, state, state_des=zeros(4)):
        self.X = self.solve(state, state_des)
        return self.X[-self.N]

    def get_u(self):
        return self.X[-self.N]




if __name__ == "__main__":

    import matplotlib.pyplot as plt

    N = 20
    dt = 0.05

    mpc = cartpole_MPC(1, 0.1, 0.5, dt=dt, N=N)

# x, theta, xdot, thetadot
    state = array([1, 0, 0, 0])
    state_des = array([0, 0, 0, 0])

    us = []
    xs = []
    thetas = []
    xdots = []
    thetadots = []


    for i in range(200):

        X = mpc.solve(state, state_des)
        u = mpc.get_u()

        # print(state)
        # state = mpc.A_dis @ state + mpc.B_dis @ u.flatten() + np.random.normal(0, )
        state = mpc.A_dis @ state + mpc.B_dis @ u.flatten()
        print(i, state)
        x, theta, xdot, thetadot = state

        # x = X[:4*N:4]
        # theta = X[1:4*N:4]
        # xdot = X[2:4*N:4]
        # thetadot = X[3:4*N:4]
        # u = X[]

        us.append(u)
        xs.append(x)
        thetas.append(theta)
        xdots.append(xdot)
        thetadots.append(thetadot)

    import csv, os
    dir = os.path.dirname(os.path.abspath(__file__))
    with open(dir+'/data.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['u', 'x', 'theta', 'xdot', 'thetadot'])
        for i in range(len(us)):
            writer.writerow([us[i], xs[i], thetas[i], xdots[i], thetadots[i]])

    



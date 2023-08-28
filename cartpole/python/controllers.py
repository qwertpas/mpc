from scipy import linalg
import numpy as np


g = 9.81

class cartpole_LQR():

    def __init__(self, M, m, L, ):
        '''
        Cart (M) with point-mass (m) on a pole (L)
        '''

        # https://danielpiedrahita.wordpress.com/portfolio/cart-pole-control/
        # state vector X = [x, theta, xdot, thetadot]
        # linearized dynamics at X = [0, 0, 0, 0]
        self.A = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, -m*g/M, 0, 0],
            [0, (m*g+M*g)/(M*L), 0, 0]
        ])
        self.B = np.array([
            [0],
            [0],
            [1/M],
            [1/(M*L)]
        ])

        self.Q = np.diag([1, 1, 1, 1])  # weight for state error
        self.R = np.eye(1)              # weight for actuation
        self.P = linalg.solve_continuous_are(self.A, self.B, self.Q, self.R)

        # self.K = np.dot(np.linalg.inv(self.R), np.dot(self.B.T, self.P))
        self.K = np.array([0.5, 0.3, 20, 1]) #???

    def get_u(self, state):
        return self.K @ state
    
class cartpole_MPC():

    def __init__(self, M, m, L, ):
        '''
        Cart (M) with point-mass (m) on a pole (L)
        '''
        A = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, -m*g/M, 0, 0],
            [0, (m*g+M*g)/(M*L), 0, 0]
        ])
        B = np.array([
            [0],
            [0],
            [1/M],
            [1/(M*L)]
        ])
        self.A

        self.Q = np.diag([1, 1, 1, 1])  # weight for state error
        self.R = np.eye(1)              # weight for actuation
        self.P = linalg.solve_continuous_are(self.A, self.B, self.Q, self.R)

        # self.K = np.dot(np.linalg.inv(self.R), np.dot(self.B.T, self.P))
        self.K = np.array([0.5, 0.3, 20, 1]) #???

    def get_u(self, state):
        return self.K @ state
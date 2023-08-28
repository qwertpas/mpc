from scipy.linalg import block_diag, solve_continuous_are, pinv, expm
import numpy as np
from numpy import array, eye, hstack, vstack, diag, zeros
np.set_printoptions(precision=0, suppress=True)
                       
g = 9.81

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

    def __init__(self, M, m, L, dt):
        '''
        Cart (M) with point-mass (m) on a pole (L)
        '''


        self.N = 2
        self.x_costs = array([1, 1, 1, 1])
        self.u_cost = 1


        # self.A = array([
        #     [0, 0, 1, 0],
        #     [0, 0, 0, 1],
        #     [0, -m*g/M, 0, 0],
        #     [0, (m*g+M*g)/(M*L), 0, 0]
        # ])
        # self.B = array([
        #     [0],
        #     [0],
        #     [1/M],
        #     [1/(M*L)]
        # ])

        self.A = 9*np.random.random((4,4))
        self.B = 9*np.random.random((4,1))
        
        # self.A_dis = expm(self.A*dt)
        # self.B_dis = pinv(self.A) @ (self.A_dis - eye(4)) @ self.B

        self.A_dis = self.A
        self.B_dis = self.B


    def get_u(self, state, state_des):

        N = self.N
        x_costs = self.x_costs
        u_cost = self.u_cost
        A_dis = self.A_dis
        B_dis = self.B_dis

        Q = diag(x_costs)
        f = -2 * state_des * x_costs
        for i in range(N): #Q becomes 4(N+1) x 4(N+1)
            Q = block_diag(Q, diag(x_costs))
            f = hstack((f, -2 * state_des * x_costs))

        R = u_cost * eye(N)
        H = block_diag(Q, R)

        f_aug = hstack((f, zeros(N)))

        A_eq = hstack((eye(4), zeros((4,4*N)), zeros((4,N))))
        for i in range(N):
            eq_constraint = hstack((zeros((4,4*i)), -A_dis, eye(4), zeros((4,4*(N-1-i)+i)), -B_dis, zeros((4,(N-1-i)))))
            A_eq = vstack((A_eq, eq_constraint))
            
        print(f"Aeq: {i} \n", A_eq, A_eq.shape, '\n')




    


if __name__ == "__main__":



    mpc = cartpole_MPC(1, 0.1, 0.5, 1/60.)

    state = array([1, 2, 3, 4])
    state_des = array([5, 6, 7, 8])

    mpc.get_u(state, state_des)

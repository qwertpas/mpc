import numpy as np
from numpy import sin, cos, pi


q = np.array(
    [0, 0.5, 1, pi / 6, pi / 4]
).T  # wheel x, pendulum angle, leg length, shoulder angle, elbow angle
qd = np.array([0, 0, 0, 0, 0]).T
m = np.array([1, 10, 2]).T  # wheel, body, hand masses
L = np.array([1, 1]).T  # upper arm, forearm lengths
N = np.array([10, 10, 10]).T  # wheel, shoulder, elbow gear ratios
I = np.array([0.1, 0.1, 0.1]).T  # wheel, body, hand structure inertias
J = np.array([0.1, 0.1, 0.1]).T  # wheel, body, hand rotor inertias
g = 9.81

M = np.matrix(
    [
        [
            J[0] * N[0] ** 2 + m[0] + m[1] + m[2],
            (
                cos(q[1] + q[3]) * L[0]
                + cos(q[1] + q[3] + q[4]) * L[1]
                - cos(q[1]) * q[2]
            )
            * m[2]
            - cos(q[1]) * m[1] * q[2],
            (-m[1] - m[2]) * sin(q[1]),
            (cos(q[1] + q[3]) * L[0] + cos(q[1] + q[3] + q[4]) * L[1]) * m[2],
            cos(q[1] + q[3] + q[4]) * L[1] * m[2],
        ],
        [
            (
                cos(q[1] + q[3]) * L[0]
                + cos(q[1] + q[3] + q[4]) * L[1]
                - cos(q[1]) * q[2]
            )
            * m[2]
            - cos(q[1]) * m[1] * q[2],
            (
                (
                    sin(q[1] + q[3]) * L[0]
                    + sin(q[1] + q[3] + q[4]) * L[1]
                    - sin(q[1]) * q[2]
                )
                ** 2
                + (
                    cos(q[1] + q[3]) * L[0]
                    + cos(q[1] + q[3] + q[4]) * L[1]
                    - cos(q[1]) * q[2]
                )
                ** 2
            )
            * m[2]
            + I[0]
            + I[1]
            + I[2]
            + m[1] * q[2] ** 2,
            (sin(q[3] + q[4]) * L[1] + sin(q[3]) * L[0]) * m[2],
            -cos(q[3] + q[4]) * L[1] * m[2] * q[2]
            - cos(q[3]) * L[0] * m[2] * q[2]
            + 2 * cos(q[4]) * L[0] * L[1] * m[2]
            + I[1]
            + I[2]
            + L[0] ** 2 * m[2]
            + L[1] ** 2 * m[2],
            -cos(q[3] + q[4]) * L[1] * m[2] * q[2]
            + cos(q[4]) * L[0] * L[1] * m[2]
            + I[2]
            + L[1] ** 2 * m[2],
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
            -cos(q[3] + q[4]) * L[1] * m[2] * q[2]
            - cos(q[3]) * L[0] * m[2] * q[2]
            + 2 * cos(q[4]) * L[0] * L[1] * m[2]
            + I[1]
            + I[2]
            + L[0] ** 2 * m[2]
            + L[1] ** 2 * m[2],
            (sin(q[3] + q[4]) * L[1] + sin(q[3]) * L[0]) * m[2],
            (
                (sin(q[1] + q[3]) * L[0] + sin(q[1] + q[3] + q[4]) * L[1]) ** 2
                + (cos(q[1] + q[3]) * L[0] + cos(q[1] + q[3] + q[4]) * L[1]) ** 2
            )
            * m[2]
            + I[1]
            + I[2]
            + J[1] * N[1] ** 2,
            cos(q[4]) * L[0] * L[1] * m[2] + I[2] + L[1] ** 2 * m[2],
        ],
        [
            cos(q[1] + q[3] + q[4]) * L[1] * m[2],
            -cos(q[3] + q[4]) * L[1] * m[2] * q[2]
            + cos(q[4]) * L[0] * L[1] * m[2]
            + I[2]
            + L[1] ** 2 * m[2],
            sin(q[3] + q[4]) * L[1] * m[2],
            cos(q[4]) * L[0] * L[1] * m[2] + I[2] + L[1] ** 2 * m[2],
            I[2] + J[2] * N[2] ** 2 + L[1] ** 2 * m[2],
        ],
    ]
)
M_inv = np.linalg.inv(M)


CG = np.matrix(
    [
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
    ]
)


print("M: \n", M)
print("M_inv: \n", M_inv)
print("C+G: \n", CG)


# M*qdd + C*qd + G

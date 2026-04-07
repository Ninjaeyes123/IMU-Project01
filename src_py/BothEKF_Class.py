import numpy as np

# ============================================================
#  Quaternion + Rotation Utilities
# ============================================================

def quat_normalize(q):
    return q / np.linalg.norm(q)

def quat_multiply(q1, q2):
    """Hamilton product q = q1 ⊗ q2."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def small_angle_quat(dtheta):
    """Convert small rotation vector to quaternion."""
    dq = np.concatenate(([1.0], 0.5 * dtheta))
    return quat_normalize(dq)

def skew(v):
    x, y, z = v
    return np.array([
        [0, -z,  y],
        [z,  0, -x],
        [-y, x,  0]
    ])

def rot_from_quat(q):
    """Rotation matrix R(q) body->world."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),       2*(x*z + y*w)],
        [2*(x*y + z*w),         1 - 2*(x*x + z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w),         2*(y*z + x*w),       1 - 2*(x*x + y*y)]
    ])

# ============================================================
#  ORIENTATION EKF (q + gyro bias)
# ============================================================

class OrientationEKF:
    """
    Error-state EKF:
      Nominal state: q (4), bg (3)
      Error state:   dtheta (3), dbg (3)
      Covariance:    6x6
    """

    def __init__(self,
                 q_init=np.array([1.,0.,0.,0.]),
                 bg_init=np.zeros(3),
                 P_init=None,
                 sigma_g=0.01,
                 sigma_bg=0.0005,
                 sigma_acc=0.5,
                 sigma_mag=1.0,
                 g_vec=np.array([0,0,-9.81]),
                 m_vec=np.array([25., 0., -45.])):

        self.q = quat_normalize(q_init)
        self.bg = bg_init.copy()

        self.P = np.eye(6)*0.01 if P_init is None else P_init.copy()

        self.sigma_g = sigma_g
        self.sigma_bg = sigma_bg
        self.R_acc = np.eye(3)*(sigma_acc**2)
        self.R_mag = np.eye(3)*(sigma_mag**2)

        self.g = g_vec
        self.m = m_vec  # µT reference field

    # --------------------------------------------------------
    #  TIME UPDATE
    # --------------------------------------------------------
    def predict(self, gyro, dt):
        omega = gyro - self.bg

        # Quaternion propagation
        w = np.linalg.norm(omega)
        if w > 1e-12:
            axis = omega / w
            angle = w * dt
            dq = np.concatenate(([np.cos(angle/2)], np.sin(angle/2)*axis))
        else:
            dq = np.array([1.,0.,0.,0.])

        self.q = quat_multiply(self.q, dq)
        self.q = quat_normalize(self.q)

        # Covariance prediction
        F = np.eye(6)
        F[0:3,0:3] += -skew(omega)*dt
        F[0:3,3:6] += -np.eye(3)*dt

        G = np.zeros((6,6))
        G[0:3,0:3] = -np.eye(3)
        G[3:6,3:6] = np.eye(3)

        Qc = np.zeros((6,6))
        Qc[0:3,0:3] = (self.sigma_g**2)*np.eye(3)
        Qc[3:6,3:6] = (self.sigma_bg**2)*np.eye(3)

        Q = G @ Qc @ G.T * dt
        self.P = F @ self.P @ F.T + Q

    # --------------------------------------------------------
    #  MEAS UPDATE: ACC + MAG
    # --------------------------------------------------------
    def update_acc_mag(self, acc, mag):
        R = rot_from_quat(self.q)

        g_pred = R.T @ self.g
        m_pred = R.T @ self.m

        y = np.concatenate((acc - g_pred,
                            mag - m_pred))

        H = np.zeros((6,6))
        H[0:3,0:3] = -skew(g_pred)
        H[3:6,0:3] = -skew(m_pred)

        Rm = np.block([
            [self.R_acc, np.zeros((3,3))],
            [np.zeros((3,3)), self.R_mag]
        ])

        S = H @ self.P @ H.T + Rm
        K = self.P @ H.T @ np.linalg.inv(S)

        dx = K @ y
        dtheta = dx[0:3]
        dbg = dx[3:6]

        dq = small_angle_quat(dtheta)
        self.q = quat_multiply(self.q, dq)
        self.q = quat_normalize(self.q)

        self.bg += dbg

        I = np.eye(6)
        self.P = (I - K @ H) @ self.P

# ============================================================
#  FULL INERTIAL EKF (p, v, q, bg, ba)
# ============================================================

class InertialNavEKF:
    """
    15-state error EKF:
      Nominal: p(3), v(3), q(4), bg(3), ba(3)
      Error:   dp, dv, dtheta, dbg, dba
    """

    def __init__(self,
                 p_init=np.zeros(3),
                 v_init=np.zeros(3),
                 q_init=np.array([1.,0.,0.,0.]),
                 bg_init=np.zeros(3),
                 ba_init=np.zeros(3),
                 P_init=None,
                 sigma_g=0.01,
                 sigma_bg=0.0005,
                 sigma_a=0.1,
                 sigma_ba=0.001,
                 g_vec=np.array([0,0,-9.81])):

        self.p = p_init.copy()
        self.v = v_init.copy()
        self.q = quat_normalize(q_init)
        self.bg = bg_init.copy()
        self.ba = ba_init.copy()

        self.P = np.eye(15)*0.01 if P_init is None else P_init.copy()

        self.sigma_g = sigma_g
        self.sigma_bg = sigma_bg
        self.sigma_a = sigma_a
        self.sigma_ba = sigma_ba

        self.g = g_vec

    # --------------------------------------------------------
    #  TIME UPDATE
    # --------------------------------------------------------
    def predict(self, gyro, acc, dt):
        omega = gyro - self.bg
        a_body = acc - self.ba

        R = rot_from_quat(self.q)
        a_world = R @ a_body + self.g

        self.p += self.v*dt + 0.5*a_world*dt*dt
        self.v += a_world*dt

        w = np.linalg.norm(omega)
        if w > 1e-12:
            axis = omega / w
            angle = w * dt
            dq = np.concatenate(([np.cos(angle/2)], np.sin(angle/2)*axis))
        else:
            dq = np.array([1.,0.,0.,0.])

        self.q = quat_multiply(self.q, dq)
        self.q = quat_normalize(self.q)

        # Covariance prediction
        F = np.eye(15)
        F[0:3,3:6] = np.eye(3)*dt
        F[3:6,6:9] = -R @ skew(a_body) * dt
        F[3:6,12:15] = -R * dt
        F[6:9,6:9] += -skew(omega)*dt
        F[6:9,9:12] += -np.eye(3)*dt

        G = np.zeros((15,12))
        G[6:9,0:3] = -np.eye(3)
        G[3:6,3:6] = R
        G[9:12,6:9] = np.eye(3)
        G[12:15,9:12] = np.eye(3)

        Qc = np.zeros((12,12))
        Qc[0:3,0:3] = (self.sigma_g**2)*np.eye(3)
        Qc[3:6,3:6] = (self.sigma_a**2)*np.eye(3)
        Qc[6:9,6:9] = (self.sigma_bg**2)*np.eye(3)
        Qc[9:12,9:12] = (self.sigma_ba**2)*np.eye(3)

        Q = G @ Qc @ G.T * dt
        self.P = F @ self.P @ F.T + Q

    # --------------------------------------------------------
    #  ZUPT UPDATE
    # --------------------------------------------------------
    def update_zupt(self, sigma_zupt=0.01):
        y = -self.v
        Rv = np.eye(3)*(sigma_zupt**2)

        H = np.zeros((3,15))
        H[:,3:6] = np.eye(3)

        S = H @ self.P @ H.T + Rv
        K = self.P @ H.T @ np.linalg.inv(S)

        dx = K @ y
        self._inject(dx)

        I = np.eye(15)
        self.P = (I - K @ H) @ self.P

    # --------------------------------------------------------
    #  GPS POSITION UPDATE
    # --------------------------------------------------------
    def update_gps(self, p_meas, sigma_gps=2.0):
        y = p_meas - self.p
        Rp = np.eye(3)*(sigma_gps**2)

        H = np.zeros((3,15))
        H[:,0:3] = np.eye(3)

        S = H @ self.P @ H.T + Rp
        K = self.P @ H.T @ np.linalg.inv(S)

        dx = K @ y
        self._inject(dx)

        I = np.eye(15)
        self.P = (I - K @ H) @ self.P

    # --------------------------------------------------------
    #  MAGNETOMETER UPDATE
    # --------------------------------------------------------
    def update_mag(self, mag, m_world, sigma_mag=1.0):
        R = rot_from_quat(self.q)
        m_pred = R.T @ m_world

        y = mag - m_pred
        Rm = np.eye(3)*(sigma_mag**2)

        H = np.zeros((3,15))
        H[:,6:9] = -skew(m_pred)

        S = H @ self.P @ H.T + Rm
        K = self.P @ H.T @ np.linalg.inv(S)

        dx = K @ y
        self._inject(dx)

        I = np.eye(15)
        self.P = (I - K @ H) @ self.P

    # --------------------------------------------------------
    #  ERROR INJECTION
    # --------------------------------------------------------
    def _inject(self, dx):
        dp = dx[0:3]
        dv = dx[3:6]
        dtheta = dx[6:9]
        dbg = dx[9:12]
        dba = dx[12:15]

        self.p += dp
        self.v += dv

        dq = small_angle_quat(dtheta)
        self.q = quat_multiply(self.q, dq)
        self.q = quat_normalize(self.q)

        self.bg += dbg
        self.ba += dba
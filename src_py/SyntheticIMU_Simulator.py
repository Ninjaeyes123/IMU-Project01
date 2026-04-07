import numpy as np

# ============================================================
#  Synthetic IMU Simulator
# ============================================================

class SyntheticIMU:
    def __init__(self,
                 dt=0.05,
                 g=np.array([0,0,-9.81]),
                 m=np.array([25., 0., -45.]),
                 sigma_g=0.01,
                 sigma_a=0.1,
                 sigma_m=0.5):

        self.dt = dt
        self.g = g
        self.m = m

        self.sigma_g = sigma_g
        self.sigma_a = sigma_a
        self.sigma_m = sigma_m

        # Initial truth state
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.q = np.array([1.,0.,0.,0.])

    # Quaternion helpers
    def quat_multiply(self, q1, q2):
        w1,x1,y1,z1 = q1
        w2,x2,y2,z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    def quat_normalize(self, q):
        return q / np.linalg.norm(q)

    def rot_from_quat(self, q):
        w,x,y,z = q
        return np.array([
            [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
            [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]
        ])

    # --------------------------------------------------------
    #  Step the simulator forward one timestep
    # --------------------------------------------------------
    def step(self, t):
        dt = self.dt

        # Synthetic trajectory: forward + gentle sinusoidal yaw
        speed = 1.2  # m/s
        yaw_rate = 0.3*np.sin(0.5*t)  # rad/s

        # True angular velocity in body frame
        omega = np.array([0., 0., yaw_rate])

        # Propagate quaternion
        w = np.linalg.norm(omega)
        if w > 1e-12:
            axis = omega / w
            angle = w * dt
            dq = np.concatenate(([np.cos(angle/2)], np.sin(angle/2)*axis))
        else:
            dq = np.array([1.,0.,0.,0.])

        self.q = self.quat_multiply(self.q, dq)
        self.q = self.quat_normalize(self.q)

        # True acceleration in world frame
        a_world = np.array([speed*0.0, 0, 0])  # no linear accel except gravity
        R = self.rot_from_quat(self.q)

        # Convert to body frame
        a_body = R.T @ (a_world + self.g)

        # Magnetometer prediction
        m_body = R.T @ self.m

        # Add noise
        gyro_meas = omega + np.random.randn(3)*self.sigma_g
        acc_meas  = a_body + np.random.randn(3)*self.sigma_a
        mag_meas  = m_body + np.random.randn(3)*self.sigma_m

        # Propagate true position/velocity
        self.v += a_world*dt
        self.p += self.v*dt

        return gyro_meas, acc_meas, mag_meas, self.p.copy(), self.q.copy()
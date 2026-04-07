class InertialNavEKF:
    def __init__(self,
                 p_init=None,
                 v_init=None,
                 q_init=None,
                 bg_init=None,
                 ba_init=None,
                 P_init=None,
                 sigma_g=0.01,
                 sigma_bg=0.0005,
                 sigma_a=0.1,
                 sigma_ba=0.001,
                 g_vec=np.array([0, 0, -9.81])):
        self.p = p_init if p_init is not None else np.zeros(3)
        self.v = v_init if v_init is not None else np.zeros(3)
        self.q = quat_normalize(q_init if q_init is not None else np.array([1., 0., 0., 0.]))
        self.bg = bg_init if bg_init is not None else np.zeros(3)
        self.ba = ba_init if ba_init is not None else np.zeros(3)

        self.P = P_init if P_init is not None else np.eye(15) * 0.01

        self.sigma_g = sigma_g
        self.sigma_bg = sigma_bg
        self.sigma_a = sigma_a
        self.sigma_ba = sigma_ba

        self.g = g_vec

    # ---------- TIME UPDATE ----------
    def predict(self, gyro_meas, acc_meas, dt):
        """
        gyro_meas: (3,) rad/s
        acc_meas:  (3,) m/s^2
        """
        # 1) Bias-corrected IMU
        omega = gyro_meas - self.bg
        a_body = acc_meas - self.ba

        R = rot_from_quat(self.q)

        # 2) Propagate nominal state
        a_world = R @ a_body + self.g

        self.p = self.p + self.v * dt + 0.5 * a_world * dt**2
        self.v = self.v + a_world * dt

        # Quaternion propagation
        omega_norm = np.linalg.norm(omega)
        if omega_norm > 1e-12:
            axis = omega / omega_norm
            angle = omega_norm * dt
            dq = np.concatenate(([np.cos(angle/2.0)], np.sin(angle/2.0)*axis))
        else:
            dq = np.array([1., 0., 0., 0.])

        self.q = quat_multiply(self.q, dq)
        self.q = quat_normalize(self.q)

        # bg, ba nominally constant

        # 3) Covariance prediction (error-state 15x15)
        F = np.eye(15)

        # Indices: δp(0:3), δv(3:6), δθ(6:9), δbg(9:12), δba(12:15)

        # δp_dot = δv
        F[0:3, 3:6] = np.eye(3) * dt

        # δv_dot ≈ -R [a_body]_x δθ - R δba
        F[3:6, 6:9] = -R @ skew(a_body) * dt
        F[3:6, 12:15] = -R * dt

        # δθ_dot ≈ -[ω]_x δθ - I δbg
        F[6:9, 6:9] += -skew(omega) * dt
        F[6:9, 9:12] += -np.eye(3) * dt

        # δbg_dot = 0
        # δba_dot = 0

        # Process noise mapping G (15x12): [gyro noise, accel noise, bg RW, ba RW]
        G = np.zeros((15, 12))
        # gyro noise -> attitude
        G[6:9, 0:3] = -np.eye(3)
        # accel noise -> velocity
        G[3:6, 3:6] = R
        # bg RW
        G[9:12, 6:9] = np.eye(3)
        # ba RW
        G[12:15, 9:12] = np.eye(3)

        Qc = np.zeros((12, 12))
        Qc[0:3, 0:3] = (self.sigma_g**2) * np.eye(3)
        Qc[3:6, 3:6] = (self.sigma_a**2) * np.eye(3)
        Qc[6:9, 6:9] = (self.sigma_bg**2) * np.eye(3)
        Qc[9:12, 9:12] = (self.sigma_ba**2) * np.eye(3)

        Q = G @ Qc @ G.T * dt

        self.P = F @ self.P @ F.T + Q

    # ---------- MEASUREMENT UPDATE: ZUPT ----------
    def update_zupt(self, R_vel):
        """
        Zero-velocity update: measurement z = 0 for velocity.
        R_vel: 3x3 measurement noise covariance for velocity.
        """
        # Measurement: v ≈ 0
        y = -self.v  # residual

        H = np.zeros((3, 15))
        H[:, 3:6] = np.eye(3)  # sensitivity to δv

        S = H @ self.P @ H.T + R_vel
        K = self.P @ H.T @ np.linalg.inv(S)

        delta_x = K @ y

        self._inject_error(delta_x)

        I = np.eye(15)
        self.P = (I - K @ H) @ self.P

    # ---------- MEASUREMENT UPDATE: GPS POSITION ----------
    def update_gps(self, p_meas, R_pos):
        """
        GPS position measurement.
        p_meas: (3,)
        R_pos:  3x3 measurement covariance
        """
        y = p_meas - self.p

        H = np.zeros((3, 15))
        H[:, 0:3] = np.eye(3)  # δp

        S = H @ self.P @ H.T + R_pos
        K = self.P @ H.T @ np.linalg.inv(S)

        delta_x = K @ y

        self._inject_error(delta_x)

        I = np.eye(15)
        self.P = (I - K @ H) @ self.P

    # ---------- MEASUREMENT UPDATE: MAG HEADING (OPTIONAL) ----------
    def update_mag(self, mag_meas, m_world, R_mag):
        """
        Magnetometer measurement.
        mag_meas: (3,) in body frame
        m_world:  (3,) reference field in world frame
        R_mag:    3x3 measurement covariance
        """
        R = rot_from_quat(self.q)
        m_body_pred = R.T @ m_world

        y = mag_meas - m_body_pred

        H = np.zeros((3, 15))
        # m_body ≈ (I - [δθ]_x) R^T m_world => δm_body ≈ -[m_body_pred]_x δθ
        H[:, 6:9] = -skew(m_body_pred)

        S = H @ self.P @ H.T + R_mag
        K = self.P @ H.T @ np.linalg.inv(S)

        delta_x = K @ y

        self._inject_error(delta_x)

        I = np.eye(15)
        self.P = (I - K @ H) @ self.P

    # ---------- ERROR INJECTION ----------
    def _inject_error(self, delta_x):
        """
        delta_x: (15,) = [δp, δv, δθ, δbg, δba]
        """
        dp = delta_x[0:3]
        dv = delta_x[3:6]
        dtheta = delta_x[6:9]
        dbg = delta_x[9:12]
        dba = delta_x[12:15]

        self.p += dp
        self.v += dv

        dq = small_angle_quat(dtheta)
        self.q = quat_multiply(self.q, dq)
        self.q = quat_normalize(self.q)

        self.bg += dbg
        self.ba += dba
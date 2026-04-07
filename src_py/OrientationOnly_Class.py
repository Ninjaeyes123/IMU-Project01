class OrientationEKF:
    def __init__(self,
                 q_init=None,
                 bg_init=None,
                 P_init=None,
                 sigma_g=0.01,
                 sigma_bg=0.0005,
                 sigma_acc=0.1,
                 sigma_mag=0.1,
                 g_vec=np.array([0, 0, -9.81]),
                 m_vec=np.array([1.0, 0.0, 0.0])):
        self.q = quat_normalize(q_init if q_init is not None else np.array([1., 0., 0., 0.]))
        self.bg = bg_init if bg_init is not None else np.zeros(3)

        self.P = P_init if P_init is not None else np.eye(6) * 0.01

        # Process noise (continuous-time)
        self.sigma_g = sigma_g
        self.sigma_bg = sigma_bg

        # Measurement noise
        self.R_acc = np.eye(3) * (sigma_acc**2)
        self.R_mag = np.eye(3) * (sigma_mag**2)

        self.g = g_vec
        self.m = m_vec

    # ---------- TIME UPDATE ----------
    def predict(self, gyro_meas, dt):
        """
        gyro_meas: measured angular rate (rad/s), shape (3,)
        """
        # 1) Bias-corrected gyro
        omega = gyro_meas - self.bg  # body frame

        # 2) Propagate quaternion (simple first-order)
        omega_norm = np.linalg.norm(omega)
        if omega_norm > 1e-12:
            axis = omega / omega_norm
            angle = omega_norm * dt
            dq = np.concatenate(([np.cos(angle/2.0)], np.sin(angle/2.0)*axis))
        else:
            dq = np.array([1., 0., 0., 0.])

        self.q = quat_multiply(self.q, dq)
        self.q = quat_normalize(self.q)

        # 3) Bias random walk: nominal bg stays same; handled via process noise in P

        # 4) Covariance prediction (error-state)
        # Error state: [δθ, δbg]
        F = np.eye(6)
        # δθ_dot ≈ -[ω]_x δθ - I δbg
        F[0:3, 0:3] += -skew(omega) * dt
        F[0:3, 3:6] += -np.eye(3) * dt
        # δbg_dot = 0

        # Process noise mapping
        G = np.zeros((6, 6))
        # gyro noise into attitude
        G[0:3, 0:3] = -np.eye(3)
        # bias RW
        G[3:6, 3:6] = np.eye(3)

        Qc = np.zeros((6, 6))
        Qc[0:3, 0:3] = (self.sigma_g**2) * np.eye(3)
        Qc[3:6, 3:6] = (self.sigma_bg**2) * np.eye(3)

        Q = G @ Qc @ G.T * dt

        self.P = F @ self.P @ F.T + Q

    # ---------- MEASUREMENT UPDATE: ACCEL ONLY ----------
    def update_acc(self, acc_meas):
        """
        acc_meas: measured acceleration (m/s^2), shape (3,)
        Assumes only gravity is present (static or slowly moving).
        """
        R = rot_from_quat(self.q)  # body->world
        g_body_pred = R.T @ self.g

        # Residual
        y = acc_meas - g_body_pred

        # Measurement Jacobian H (3x6) wrt error state [δθ, δbg]
        # g_body ≈ (I - [δθ]_x) R^T g  => δg_body ≈ -[R^T g]_x δθ
        H = np.zeros((3, 6))
        H[:, 0:3] = -skew(g_body_pred)
        # no direct sensitivity to bg
        # H[:, 3:6] = 0

        S = H @ self.P @ H.T + self.R_acc
        K = self.P @ H.T @ np.linalg.inv(S)

        delta_x = K @ y  # (6,)

        # Inject error into nominal state
        dtheta = delta_x[0:3]
        dbg = delta_x[3:6]

        dq = small_angle_quat(dtheta)
        self.q = quat_multiply(self.q, dq)
        self.q = quat_normalize(self.q)

        self.bg += dbg

        # Covariance update
        I = np.eye(6)
        self.P = (I - K @ H) @ self.P

    # ---------- MEASUREMENT UPDATE: ACC + MAG ----------
    def update_acc_mag(self, acc_meas, mag_meas):
        """
        Joint update using accelerometer and magnetometer.
        """
        R = rot_from_quat(self.q)

        g_body_pred = R.T @ self.g
        m_body_pred = R.T @ self.m

        y = np.concatenate((acc_meas - g_body_pred,
                            mag_meas - m_body_pred))

        H = np.zeros((6, 6))
        H[0:3, 0:3] = -skew(g_body_pred)
        H[3:6, 0:3] = -skew(m_body_pred)

        R_meas = np.block([
            [self.R_acc, np.zeros((3, 3))],
            [np.zeros((3, 3)), self.R_mag]
        ])

        S = H @ self.P @ H.T + R_meas
        K = self.P @ H.T @ np.linalg.inv(S)

        delta_x = K @ y
        dtheta = delta_x[0:3]
        dbg = delta_x[3:6]

        dq = small_angle_quat(dtheta)
        self.q = quat_multiply(self.q, dq)
        self.q = quat_normalize(self.q)

        self.bg += dbg

        I = np.eye(6)
        self.P = (I - K @ H) @ self.P
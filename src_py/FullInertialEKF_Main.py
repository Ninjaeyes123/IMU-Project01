ekf = InertialNavEKF()

for gyro, acc, mag in imu_stream:
    ekf.predict(gyro, acc, dt)

    # If foot stationary:
    # ekf.update_zupt()

    # If GPS available:
    # ekf.update_gps(gps_pos)

    # If using magnetometer:
    # ekf.update_mag(mag, m_world)
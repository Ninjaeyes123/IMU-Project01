dt = 1/20

ekf = OrientationEKF()

for gyro, acc, mag in imu_stream:
    ekf.predict(gyro, dt)
    ekf.update_acc_mag(acc, mag)

    print("Orientation (quat):", ekf.q)
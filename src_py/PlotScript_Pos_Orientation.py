import numpy as np
import matplotlib.pyplot as plt

def quat_to_euler(q):
    """Convert quaternion to roll, pitch, yaw."""
    w,x,y,z = q
    # roll
    r = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    # pitch
    p = np.arcsin(2*(w*y - z*x))
    # yaw
    y_ = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return np.array([r,p,y_])

def run_sim_and_plot():
    dt = 0.05
    sim = SyntheticIMU(dt=dt)
    ekf = OrientationEKF()

    T = 20.0
    steps = int(T/dt)

    pos_true = []
    pos_est  = []
    eul_true = []
    eul_est  = []

    t = 0.0
    for k in range(steps):
        gyro, acc, mag, p_true, q_true = sim.step(t)

        # EKF
        ekf.predict(gyro, dt)
        ekf.update_acc_mag(acc, mag)

        pos_true.append(p_true)
        pos_est.append(np.zeros(3))  # orientation-only EKF has no position
        eul_true.append(quat_to_euler(q_true))
        eul_est.append(quat_to_euler(ekf.q))

        t += dt

    pos_true = np.array(pos_true)
    eul_true = np.array(eul_true)
    eul_est  = np.array(eul_est)

    # --------------------------------------------------------
    #  Plot orientation
    # --------------------------------------------------------
    plt.figure(figsize=(12,6))
    labels = ["Roll", "Pitch", "Yaw"]
    for i in range(3):
        plt.subplot(3,1,i+1)
        plt.plot(eul_true[:,i], label="True")
        plt.plot(eul_est[:,i], '--', label="EKF")
        plt.ylabel(labels[i])
        if i == 0:
            plt.legend()
    plt.xlabel("Time step")
    plt.suptitle("Orientation (Euler angles)")
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    #  Plot position (true only)
    # --------------------------------------------------------
    plt.figure(figsize=(10,5))
    plt.plot(pos_true[:,0], pos_true[:,1], label="True trajectory")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("True Position (EKF orientation-only has no position)")
    plt.legend()
    plt.axis("equal")
    plt.show()

# Run it
if __name__ == "__main__":
    run_sim_and_plot()
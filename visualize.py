#!/usr/bin/env python3

from uav_simulator import UAV_simulator
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Initialize simulator
    x0 = np.array([[0], [0], [0]])
    rpy = np.array([np.deg2rad(0), np.deg2rad(0), np.deg2rad(0)])
    sim = UAV_simulator(x0, rpy, follow=True)

    # example of cycling through UAV orientations
    print("Displaying UAV Orientation Animation Sample...")
    for i in range(x0[0, 0], x0[0, 0] + 20, 1):
        sim.UpdateRPY(np.array((np.deg2rad(i), np.deg2rad(0), np.deg2rad(0))))
    for j in range(x0[1, 0], x0[1, 0] + 20, 1):
        sim.UpdateRPY(np.array((np.deg2rad(i), np.deg2rad(j), np.deg2rad(0))))
    for k in range(x0[2, 0], x0[2, 0] + 20, 1):
        sim.UpdateRPY(np.array((np.deg2rad(i), np.deg2rad(j), np.deg2rad(k))))

    # example of cycling through UAV positions
    print("Displaying UAV Position Animation Sample...")
    for i in range(0, 10, 1):
        sim.UpdateX(np.array((i, 0, 0)))
    for j in range(0, 10, 1):
        sim.UpdateX(np.array((i, j, 0)))
    for k in range(0, 10, 1):
        sim.UpdateX(np.array((i, j, k)))

    plt.show()
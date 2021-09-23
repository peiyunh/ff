# sampler.py
# trajectory sampler
# including a mix of clothoids, straight lines, and circles
import numpy as np
from scipy.special import fresnel

def sample(v0, Kappa, T0, N0, tt, M, debug=False):
    # sample accelerations
    if debug:
        accelerations = np.full(M, 0)
    else:
        accelerations = 10*(np.random.rand(M)-0.5)  # -5m/s^2 to 5m/s^2

    # sample velocities
    if debug:  # no randomness in velocity
        velocities = np.full(M, v0)
    else:  # randomly sample a velocity <=15m/s at 80% of time
        v_options = np.stack((np.full(M, v0), 15*np.random.rand(M)))
        v_selections = (np.random.rand(M) >= 0.2).astype(int)
        velocities = v_options[v_selections, np.arange(M)]

    # generate longitudinal distances
    L = velocities[:, None] * tt[None, :] + accelerations[:, None] * (tt[None, :]**2) / 2
    # print("L:", L)

    #
    if debug:
        alphas = np.full(M, v0)
    else:
        alphas = (80 - 6) * np.random.rand(M) + 6

    ############################################################################
    # sample M straight lines
    line_points = L[:, :, None] * T0[None, None, :]
    line_thetas = np.zeros_like(L)
    lines = np.concatenate((line_points, line_thetas[:, :, None]), axis=-1)

    ############################################################################
    # sample M circles
    Krappa = min(-0.01, Kappa) if Kappa <= 0 else max(0.01, Kappa)
    # print("Krappa:", Krappa)

    radius = np.abs(1 / Krappa)
    # print("radius:", radius)

    center = np.array([-1 / Krappa, 0])
    # print("center:", center)

    circle_phis = L / radius if Krappa >= 0 else np.pi - L/radius
    # print("circle_phis:", circle_phis)

    circle_points = np.dstack([
        center[0] + radius * np.cos(circle_phis),
        center[1] + radius * np.sin(circle_phis),
    ])
    # print("circle_points:", circle_points)

    # rotate thetas
    circle_thetas = L/radius if Krappa >= 0 else -L/radius
    # wrap
    circle_thetas = (circle_thetas + np.pi) % (2 * np.pi) - np.pi
    #
    circles = np.concatenate((circle_points, circle_thetas[:, :, None]), axis=-1)

    ############################################################################
    # sample M clothoids
    # TODO figure out how to do this ...
    # NOTE
    #   the way I was sampling clothoids was wrong, leading to sampling weird turning motions
    #   I am trying to correct that based on clothoid.py
    # Xi0 = Kappa / np.pi
    # Xis = Xi0 + L
    Xi0 = np.abs(Kappa) / np.pi
    Xis = Xi0 + L

    #
    # Ss, Cs = fresnel((Xis - Xi0) / alphas[:, None])
    Ss, Cs = fresnel(Xis / alphas[:, None])

    clothoid_points = alphas[:, None, None] * (Cs[:, :, None]*T0[None, None, :] + Ss[:, :, None]*N0[None, None, :])
    # print("clothoid_points:", clothoid_points)

    # print("X0, Y0:", clothoid_points[:,0,:])

    #
    Xs = clothoid_points[:, :, 0] - clothoid_points[:, 0, 0, None]
    Ys = clothoid_points[:, :, 1] - clothoid_points[:, 0, 1, None]
    clothoid_theta0s = 0.5 * np.pi * ((Kappa / np.pi / alphas) ** 2)
    clothoid_theta0s = clothoid_theta0s[:, None]
    signed_clothoid_theta0s = clothoid_theta0s * np.sign(Kappa)
    # when kappa is positive, the clothoid curves left, theta is positive
    # we will rotate it clockwise by theta
    # when kappa is negative, the clothoid curves right, theta is negative
    # we will rotate it counterclockwise by theta
    clothoid_points[:, :, 0] = np.cos(signed_clothoid_theta0s) * Xs + np.sin(signed_clothoid_theta0s) * Ys
    clothoid_points[:, :, 1] = - np.sin(signed_clothoid_theta0s) * Xs + np.cos(signed_clothoid_theta0s) * Ys

    # tangent vector: http://mathworld.wolfram.com/CornuSpiral.html
    clothoid_thetas = 0.5 * np.pi * ((Xis / alphas[:, None])**2)
    clothoid_thetas = clothoid_thetas - clothoid_theta0s
    signed_clothoid_thetas = clothoid_thetas * np.sign(Kappa)
    # clothoid_thetas = clothoid_thetas if Krappa >= 0 else -clothoid_thetas
    # wrap
    # clothoid_thetas = (clothoid_thetas + np.pi) % (2 * np.pi) - np.pi
    wrapped_signed_clothoid_thetas = (signed_clothoid_thetas + np.pi) % (2 * np.pi) - np.pi
    # wrapped_signed_clothoid_thetas = (signed_clothoid_thetas) % (2 * np.pi)
    #
    clothoids = np.concatenate((clothoid_points, wrapped_signed_clothoid_thetas[:, :, None]), axis=-1)

    ############################################################################
    # pick M in total
    t_options = np.stack((lines, circles, clothoids))
    t_selections = np.random.choice([0, 1, 2], size=M, p=(0.5, 0.25, 0.25))

    if debug:
        t_selections[:] = 2

    ############################################################################
    # randomly mirroring
    # trajectories = t_options[t_selections, np.arange(M)]
    # signs = np.random.choice([0,1], M, p=[0.5, 0.5])
    # trajectories[:,:,0] *= signs[:,None]

    #
    trajs = t_options[t_selections, np.arange(M)]

    # toss a coin for vertical flipping
    heads = (np.random.rand(M) <= 0.5)
    tails = np.logical_not(heads)

    # NOTE theta means what here
    conditions = [heads[:, None, None], tails[:, None, None]]
    choices = [trajs, np.dstack((
        -trajs[:, :, 0], trajs[:, :, 1], -trajs[:, :, 2]
    ))]

    trajectories = np.select(conditions, choices)

    if debug:
        trajectories = trajs

    # return trajectories, t_selections

    return trajectories


if __name__ == "__main__":
    from nuscenes.nuscenes import NuScenes
    from nuscenes.can_bus.can_bus_api import NuScenesCanBus
    import matplotlib.pyplot as plt
    nusc = NuScenes("v1.0-mini", "/data/nuscenes")
    nusc_can = NuScenesCanBus(dataroot="/data/nuscenes")

    for scene in nusc.scene:
        scene_name = scene["name"]
        scene_id = int(scene_name[-4:])
        if scene_id in nusc_can.can_blacklist:
            print(f"skipping {scene_name}")
            continue
        pose = nusc_can.get_messages(scene_name, "pose")
        saf = nusc_can.get_messages(scene_name, "steeranglefeedback")
        vm = nusc_can.get_messages(scene_name, "vehicle_monitor")
        # NOTE: I tried to verify if the relevant measurements are consistent
        # across multiple tables that contain redundant information
        # NOTE: verified pose's velocity matches vehicle monitor's
        # but the pose table offers at a much higher frequency
        # NOTE: same that steeranglefeedback's steering angle matches vehicle monitor's
        # but the steeranglefeedback table offers at a much higher frequency
        print(pose[23])
        print(saf[45])
        print(vm[0])

        # initial velocity (m/s)
        v0 = pose[23]["vel"][0]
        # curvature
        Kappa = 2 * saf[45]["value"] / 2.588
        # T0: longitudinal axis
        T0 = np.array([0.0, 1.0])
        # N0: normal directional vector
        N0 = np.array([1.0, 0.0]) if Kappa <= 0 else np.array([-1.0, 0.0])
        # tt: time stamps
        tt = np.arange(0.0, 3.01, 0.01)
        # M: number of samples
        M = 2000
        #
        debug = False
        #
        trajectories = sample(v0, Kappa, T0, N0, tt, M, debug)
        #
        for i in range(len(trajectories)):
            trajectory = trajectories[i]
            plt.plot(trajectory[:, 0], trajectory[:, 1])
        plt.grid(False)
        plt.axis("equal")

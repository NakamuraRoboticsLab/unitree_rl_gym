import time

import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml
import onnxruntime as ort
import os
import sys

# make top-level 'deploy' importable so we can reach deploy.deploy_real.common
sys.path.append(LEGGED_GYM_ROOT_DIR)
sys.path.append('/home/zewenhe/src/unitree_rl_gym')
sys.path.append('/home/zewenhe/src/unitree_rl_gym/deploy/deploy_real/common')

from deploy.deploy_real.common.rotation_helper import transform_pelvis_to_torso_complete


def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


# ===== Helpers for bydmimic policy (ported from deploy_real4bydmimic.py) =====

joint_seq = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]

joint_xml = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


def quaternion_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.array([w, x, y, z])


def quaternion_to_rotation_matrix(q):
    q = np.array(q, dtype=np.float64)
    q = q / np.linalg.norm(q)

    w, x, y, z = q

    r00 = 1 - 2 * y ** 2 - 2 * z ** 2
    r01 = 2 * x * y - 2 * z * w
    r02 = 2 * x * z + 2 * y * w

    r10 = 2 * x * y + 2 * z * w
    r11 = 1 - 2 * x ** 2 - 2 * z ** 2
    r12 = 2 * y * z - 2 * x * w

    r20 = 2 * x * z - 2 * y * w
    r21 = 2 * y * z + 2 * x * w
    r22 = 1 - 2 * x ** 2 - 2 * y ** 2

    return np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]])


def yaw_quat(q):
    w, x, y, z = q
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y ** 2 + z ** 2))
    return np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])


def matrix_to_quaternion_simple(matrix):
    matrix = np.array(matrix)
    m00, m01, m02 = matrix[0]
    m10, m11, m12 = matrix[1]
    m20, m21, m22 = matrix[2]

    trace = m00 + m11 + m22

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m21 - m12) * s
        y = (m02 - m20) * s
        z = (m10 - m01) * s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s

    return np.array([w, x, y, z])


if __name__ == "__main__":
    # get config file name from command line
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()
    config_file = args.config_file
    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
    xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

    simulation_duration = config["simulation_duration"]
    simulation_dt = config["simulation_dt"]
    control_decimation = config["control_decimation"]

    num_actions = config["num_actions"]
    num_obs = config["num_obs"]

    mode = config.get("mode", "walk")

    if mode == "bydmimic":
        # gains and defaults in 29-dof actuator order
        kps = np.array(config["stiffness"], dtype=np.float32)
        kds = np.array(config["damping"], dtype=np.float32)
        default_angles = np.array(config["default_angles"], dtype=np.float32)

        default_angles_seq = np.array(config["default_angles_seq"], dtype=np.float32)
        action_scale_seq = np.array(config["action_scale_seq"], dtype=np.float32)

        # fixed cmd and scales kept only for structure compatibility
        cmd = np.array(config.get("cmd_init", [0.0, 0.0, 0.0]), dtype=np.float32)

    else:
        # original walking policy settings
        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        cmd = np.array(config["cmd_init"], dtype=np.float32)

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0
    timestep = 0

    # used only in bydmimic mode
    action_buffer = np.zeros((num_actions,), dtype=np.float32)
    init_to_world = np.zeros((3, 3), dtype=np.float32)

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # load policy
    if mode == "bydmimic":
        policy = ort.InferenceSession(policy_path)
        # Debug: check ONNX input / output names and shapes
        print("[bydmimic] ONNX inputs:", [(i.name, i.shape) for i in policy.get_inputs()])
        print("[bydmimic] ONNX outputs:", [(o.name, o.shape) for o in policy.get_outputs()])
    else:
        policy = torch.jit.load(policy_path)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            if mode == "bydmimic":
                # q and dq in actuator joint order using names
                q = np.zeros(num_actions, dtype=np.float32)
                dq = np.zeros(num_actions, dtype=np.float32)
                for i, name in enumerate(joint_xml):
                    j_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
                    qadr = m.jnt_qposadr[j_id]
                    dadr = m.jnt_dofadr[j_id]
                    q[i] = d.qpos[qadr]
                    dq[i] = d.qvel[dadr]
                tau = pd_control(target_dof_pos, q, kps, np.zeros_like(kds), dq, kds)
                d.ctrl[:] = 0.0
                # first 29 actuators correspond to joint_xml order
                d.ctrl[:num_actions] = tau
            else:
                tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
                d.ctrl[:] = tau
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)

            counter += 1
            if counter % control_decimation == 0:
                if mode == "bydmimic":
                    # ===== observation for bydmimic policy =====
                    # from common.rotation_helper import transform_pelvis_to_torso_complete

                    # joint positions / velocities in actuator joint order
                    qj = np.zeros(num_actions, dtype=np.float32)
                    dqj = np.zeros(num_actions, dtype=np.float32)
                    for i, name in enumerate(joint_xml):
                        j_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
                        qadr = m.jnt_qposadr[j_id]
                        dadr = m.jnt_dofadr[j_id]
                        qj[i] = d.qpos[qadr]
                        dqj[i] = d.qvel[dadr]

                    # base orientation and angular velocity (pelvis frame)
                    quat_pelvis = d.qpos[3:7]
                    omega = d.qvel[3:6]

                    # torso orientation from pelvis imu and waist joints
                    waist_yaw_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "waist_yaw_joint")
                    waist_roll_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "waist_roll_joint")
                    waist_pitch_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "waist_pitch_joint")
                    waist_yaw = d.qpos[m.jnt_qposadr[waist_yaw_id]]
                    waist_roll = d.qpos[m.jnt_qposadr[waist_roll_id]]
                    waist_pitch = d.qpos[m.jnt_qposadr[waist_pitch_id]]

                    quat_torso = transform_pelvis_to_torso_complete(waist_yaw, waist_roll, waist_pitch, quat_pelvis)

                    # load motion data once lazily
                    if "_motion_cache" not in globals():
                        globals()["_motion_cache"] = np.load(
                            f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/bydmimic/dance1.npz"
                        )
                    motion = globals()["_motion_cache"]
                    motionpos = motion["body_pos_w"]
                    motionquat = motion["body_quat_w"]
                    motioninputpos = motion["joint_pos"]
                    motioninputvel = motion["joint_vel"]

                    if timestep < 2:
                        ref_motion_quat = motionquat[timestep, 9, :]
                        yaw_motion_quat = yaw_quat(ref_motion_quat)
                        yaw_motion_matrix = quaternion_to_rotation_matrix(yaw_motion_quat)

                        robot_quat = quat_torso
                        yaw_robot_quat = yaw_quat(robot_quat)
                        yaw_robot_matrix = quaternion_to_rotation_matrix(yaw_robot_quat)
                        init_to_world = yaw_robot_matrix @ yaw_motion_matrix.T

                    qj_obs = qj.copy()
                    dqj_obs = dqj.copy()

                    # ------------------------------------------------------------------
                    # Observation construction aligned with training-time PolicyCfg:
                    #   [command,
                    #    motion_anchor_pos_b, motion_anchor_ori_b,
                    #    base_lin_vel, base_ang_vel,
                    #    joint_pos_rel, joint_vel_rel,
                    #    actions]
                    # ------------------------------------------------------------------

                    # command term: reference motion joint pos/vel
                    cmd_joint_dim = motioninputpos.shape[1]
                    command = np.concatenate(
                        (motioninputpos[timestep, :], motioninputvel[timestep, :]), axis=0
                    )

                    # motion anchor (from reference motion) and robot anchor (from Mujoco)
                    # here we use body index 9 from the motion file as anchor, consistent
                    # with how torso is used in the original real deployment.
                    anchor_pos_w = motionpos[timestep, 9, :]
                    anchor_quat_w = motionquat[timestep, 9, :]

                    # robot anchor: use base position and torso orientation in world frame
                    robot_anchor_pos_w = d.qpos[0:3].copy()
                    robot_anchor_quat_w = quat_torso

                    # relative transform from robot anchor frame to motion anchor frame
                    R_robot = quaternion_to_rotation_matrix(robot_anchor_quat_w)
                    motion_anchor_pos_b = R_robot.T @ (anchor_pos_w - robot_anchor_pos_w)

                    rel_quat = quaternion_multiply(
                        quaternion_conjugate(robot_anchor_quat_w), anchor_quat_w
                    )
                    rel_quat = rel_quat / np.linalg.norm(rel_quat)
                    rel_mat = quaternion_to_rotation_matrix(rel_quat)
                    motion_anchor_ori_b = rel_mat[:, :2].reshape(-1,)

                    # base linear / angular velocity in world frame
                    base_lin_vel = d.qvel[0:3].copy()
                    base_ang_vel = omega

                    # reorder q, dq to policy joint order (joint_pos_rel, joint_vel_rel)
                    qpos_urdf = qj_obs
                    dqpos_urdf = dqj_obs
                    qj_obs_seq = np.array([qpos_urdf[joint_xml.index(j)] for j in joint_seq])
                    dqj_obs_seq = np.array([dqpos_urdf[joint_xml.index(j)] for j in joint_seq])

                    joint_pos_rel = qj_obs_seq - default_angles_seq
                    joint_vel_rel = dqj_obs_seq

                    # fill observation vector in the exact order used in training
                    offset = 0
                    obs[offset : offset + command.shape[0]] = command
                    offset += command.shape[0]

                    obs[offset : offset + 3] = motion_anchor_pos_b
                    offset += 3

                    obs[offset : offset + 6] = motion_anchor_ori_b
                    offset += 6

                    obs[offset : offset + 3] = base_lin_vel
                    offset += 3

                    obs[offset : offset + 3] = base_ang_vel
                    offset += 3

                    obs[offset : offset + num_actions] = joint_pos_rel
                    offset += num_actions

                    obs[offset : offset + num_actions] = joint_vel_rel
                    offset += num_actions

                    obs[offset : offset + num_actions] = action_buffer
                    offset += num_actions

                    # quick consistency check (only print once if mismatch)
                    if offset != num_obs and timestep == 0:
                        print(
                            "[bydmimic] Warning: obs length mismatch:",
                            "filled =", offset,
                            "expected =", num_obs,
                        )

                    # Debug: inspect observation content and shapes in early timesteps
                    if timestep < 20:
                        print("[bydmimic] timestep", timestep)
                        print("  obs shape:", obs.shape)
                        print("  obs has nan:", np.isnan(obs).any())
                        print("  obs min/max/mean:", obs.min(), obs.max(), obs.mean())
                        print("  command shape:", command.shape)
                        print("  motion_anchor_pos_b shape:", motion_anchor_pos_b.shape)
                        print("  motion_anchor_ori_b shape:", motion_anchor_ori_b.shape)
                        print("  base_lin_vel shape:", base_lin_vel.shape)
                        print("  base_ang_vel shape:", base_ang_vel.shape)
                        print("  qj_obs_seq shape:", qj_obs_seq.shape)
                        print("  dqj_obs_seq shape:", dqj_obs_seq.shape)
                        print("  action_buffer shape:", action_buffer.shape)

                    # Save first 50 steps obs for offline comparison
                    if timestep < 50:
                        np.save(f"/tmp/obs_mujoco_{timestep}.npy", obs.copy())

                    # policy inference (ONNX)
                    obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                    act = policy.run(
                        ["actions"],
                        {
                            "obs": obs_tensor.numpy(),
                            "time_step": np.array([[timestep]], dtype=np.float32),
                        },
                    )[0]

                    act = np.asarray(act).reshape(-1)
                    action = act.copy()
                    action_buffer = act.copy()

                    target_dof_pos = default_angles_seq + action * action_scale_seq
                    target_dof_pos = target_dof_pos.reshape(-1,)
                    target_dof_pos = np.array([target_dof_pos[joint_seq.index(j)] for j in joint_xml])

                    timestep += 1
                else:
                    # ===== original walking observation and policy =====
                    # create observation
                    qj = d.qpos[7:]
                    dqj = d.qvel[6:]
                    quat = d.qpos[3:7]
                    omega = d.qvel[3:6]

                    qj = (qj - default_angles) * dof_pos_scale
                    dqj = dqj * dof_vel_scale
                    gravity_orientation = get_gravity_orientation(quat)
                    omega = omega * ang_vel_scale

                    period = 0.8
                    count = counter * simulation_dt
                    phase = count % period / period
                    sin_phase = np.sin(2 * np.pi * phase)
                    cos_phase = np.cos(2 * np.pi * phase)

                    obs[:3] = omega
                    obs[3:6] = gravity_orientation
                    obs[6:9] = cmd * cmd_scale
                    obs[9 : 9 + num_actions] = qj
                    obs[9 + num_actions : 9 + 2 * num_actions] = dqj
                    obs[9 + 2 * num_actions : 9 + 3 * num_actions] = action
                    obs[9 + 3 * num_actions : 9 + 3 * num_actions + 2] = np.array([sin_phase, cos_phase])
                    obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                    # policy inference
                    action = policy(obs_tensor).detach().numpy().squeeze()
                    # transform action to target_dof_pos
                    target_dof_pos = action * action_scale + default_angles

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

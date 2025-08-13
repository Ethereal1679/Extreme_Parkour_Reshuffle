import time

import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import yaml

from rsl_rl import modules
from rsl_rl.modules.depth_backbone import RecurrentDepthBackbone, DepthOnlyFCBackbone58x87

import cv2
import os
from collections import deque
import glfw
import cv2

SIM2SIM = os.path.dirname(os.path.abspath(__file__))
print("SIM2SIM",SIM2SIM)

## ==== functional tools ====

def resize2d(img, size):
    # import ipdb;ipdb.set_trace()
    return (F.adaptive_avg_pool2d(Variable(torch.from_numpy(img.copy())), size)).data ## 平均池化

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


def euler_from_quaternion(quat_angle):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        # x = quat_angle[:,0]; y = quat_angle[:,1]; z = quat_angle[:,2]; w = quat_angle[:,3]
        w = quat_angle[0]; x = quat_angle[1]; y = quat_angle[2]; z = quat_angle[3]
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = np.arctan2(t0, t1)
        t2 = +2.0 * (w * y - z * x)
        t2 = np.clip(t2, -1, 1)
        pitch_y = np.arcsin(t2)
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = np.arctan2(t3, t4)
    
        return roll_x, pitch_y, yaw_z # in radians





if __name__ == "__main__":
    # get config file name from command line
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()
    # config_file = args.config_file
    ## ==== configs ====
    with open(f"{SIM2SIM}/configs/go2.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

        policy_path = config["policy_path"].replace("{SIM2SIM}", SIM2SIM) ##路径替换
        vision_policy_path = config["vision_policy_path"].replace("{SIM2SIM}", SIM2SIM)
        xml_path = config["xml_path"].replace("{SIM2SIM}", SIM2SIM)

        ## camera
        output_resolution = config["output_resolution"]
        cropping = config["cropping"]
        depth_range = config["depth_range"]

        ## simulate
        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]

        n_proprio = config["n_proprio"]
        n_hist_len = config["n_hist_len"]
        n_depth_latent = config["n_depth_latent"]
        
        cmd = np.array(config["cmd_init"], dtype=np.float32)





    # ==== 初始化变量 ====
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs_prop = np.zeros(n_proprio, dtype=np.float32)


    history_obs_prop = np.zeros((n_hist_len , n_proprio), dtype=np.float32)  
    # history_obs_prop = deque(maxlen=n_hist_len)
    # history_obs_prop.clear()
    counter = 0



    # ==== 创建OpenGL上下文（离屏渲染）====
    resolution = (640, 480)
    glfw.init()
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    window = glfw.create_window(resolution[0], resolution[1], "Offscreen", None, None)
    glfw.make_context_current(window)


    # ==== robot init ====
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    scene = mujoco.MjvScene(m, maxgeom=10000)
    context = mujoco.MjrContext(m, mujoco.mjtFontScale.mjFONTSCALE_150.value)
    # qpos = d.qpos[7:7+12].copy() # 关节位置
    # qvel = d.qvel[6:6+12].copy() # 关节速度


    # ==== set camera properties ===
    camera_name = "depth_camera"
    camera_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    camera = mujoco.MjvCamera()
    ## 相机固定、跟踪
    camera.type = mujoco.mjtCamera.mjCAMERA_FIXED 
    if camera_id != -1:
        print("camera_id", camera_id)
        camera.fixedcamid = camera_id


    # ==== 创建帧缓冲对象并启用离屏渲染 ====
    framebuffer = mujoco.MjrRect(0, 0, resolution[0], resolution[1])
    mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, context)


    # ======================== local functions start ========================
    ## depth
    def encode_depth(depth_image, proprio):
        depth_latent_yaw = depth_encoder(depth_image, proprio)
        if torch.isnan(depth_latent_yaw).any():
            print('depth_latent_yaw contains nan and the depth image is: ', depth_image)
        return depth_latent_yaw


    ## actor
    def actor_model(obs):
        action = actor(obs)
        return action


    ## 组帧
    def combine_all_obs(proprio, depth_latent_yaw, proprio_history, n_proprio, n_depth_latent, n_hist_len):
        depth_latent = depth_latent_yaw[:, :-2]
        yaw = depth_latent_yaw[:, -2:] * 1.5
        print('yaw: ', yaw)
        proprio[:, 6:8] = yaw
        lin_vel_latent = estimator(proprio)
        activation = nn.ELU()
        # import ipdb;ipdb.set_trace()
        # priv_latent = hist_encoder(activation, proprio_history.view(-1, n_hist_len, n_proprio))
        proprio_history = torch.from_numpy(proprio_history).to(device).unsqueeze(0)
        priv_latent = hist_encoder(activation, proprio_history)
        # import ipdb;ipdb.set_trace()
        obs = torch.cat([proprio, depth_latent, lin_vel_latent, priv_latent], dim=-1)
        return obs


    ## 获取深度相机图片
    def _get_depth_image():
        # 更新场景
        viewport = mujoco.MjrRect(0, 0, resolution[0], resolution[1])
        mujoco.mjv_updateScene(m, d, mujoco.MjvOption(), 
                            mujoco.MjvPerturb(), camera, 
                            mujoco.mjtCatBit.mjCAT_ALL, scene)

        # 渲染场景并读取像素数据（RGB）
        mujoco.mjr_render(viewport, scene, context)
        rgb = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
        depth_buffer = np.zeros((resolution[1], resolution[0], 1), dtype=np.float32) ## 深度图像
        # fetch depth image
        mujoco.mjr_readPixels(rgb, depth_buffer, viewport, context)
        depth_buffer = np.flip(depth_buffer, axis=0).squeeze() ## 需要和OPENGL里面的颠倒方向，不然方向是反的

        # bgr = cv2.cvtColor(np.flipud(rgb), cv2.COLOR_RGB2BGR)
        # cv2.imshow('MuJoCo Camera Output', bgr)
        return depth_buffer


    ## 处理深度图像
    def _process_depth_image(depth_image):
        # 裁剪
        use_cropping = False
        if use_cropping:
            depth_image = depth_image[
                cropping[0]: -cropping[1]-1,
                cropping[2]: -cropping[3]-1,
            ]
        
        
        # AVG池化
        use_avg_pool = True
        if use_avg_pool:
            reshaped_depth_image = resize2d(depth_image[None,None,...], output_resolution)
            reshaped_depth_image = reshaped_depth_image.squeeze(0).squeeze(0).numpy() # [58 ,87]
            # reshaped_depth_image = reshaped_depth_image.squeeze(0).numpy()
            # print(reshaped_depth_image.shape)
            # import ipdb;ipdb.set_trace()
        
        # NOTE！！！ 原始的深度图，只用于cv可视化 [480,640]
        depth_image_origin = (depth_image - np.min(depth_image)) / (np.max(depth_image) - np.min(depth_image)) 
        # import ipdb;ipdb.set_trace()
        depth_image_origin = np.uint8(depth_image_origin * 255)
        
        
        # 归一化 [58,87]
        depth_normalized = (reshaped_depth_image - np.min(reshaped_depth_image)) / (np.max(reshaped_depth_image) - np.min(reshaped_depth_image)) - 0.5
        # import ipdb;ipdb.set_trace()
        # depth_normalized = np.float32(np.uint8(depth_normalized * 255))

        # 转成Tensor类型
        depth_normalized_to_tensor = torch.from_numpy(depth_normalized).unsqueeze(0).to(device)



        cv2.imshow('Depth Map (Grayscale)', depth_image_origin)
        
        ## 返回cv图  归一化深度图  tensor的归一化深度图
        return depth_image_origin, depth_normalized, depth_normalized_to_tensor
    # ======================== local functions end ========================


    # ==== warm up ====
    # depth_image = _get_depth_image() ## 获得深度图像
    # depth_normalized = _process_depth_image(depth_image) ## 处理深度图像




    # ==== 导入模型 ====
    device = "cuda"
    policy = torch.jit.load(policy_path, map_location=device)
    policy.eval()
    estimator = policy.estimator.estimator
    hist_encoder = policy.actor.history_encoder
    actor = policy.actor.actor_backbone


    vision_model = torch.load(vision_policy_path, map_location=device)
    depth_backbone = DepthOnlyFCBackbone58x87(None, 32, None)
    depth_encoder = RecurrentDepthBackbone(depth_backbone, None, eval_prop_n=n_proprio ).to(device)
    depth_encoder.load_state_dict(vision_model['depth_encoder_state_dict'])
    depth_encoder.to(device)
    depth_encoder.eval()





    ### ========== 主函数 ==========
    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()

            tau = pd_control(target_dof_pos, d.qpos[7:7+12].copy(), kps, np.zeros_like(kds), d.qvel[6:6+12].copy(), kds)
            # import ipdb;ipdb.set_trace()

            d.ctrl[:] = tau
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)


            if counter % control_decimation == 0:
                # Apply control signal here.

                ## create observation
                qj = d.qpos[7:7+12]
                dqj = d.qvel[6:6+12]
                quat = d.qpos[3:7]
                omega = d.qvel[3:6]
                gravity_orientation = get_gravity_orientation(quat) 

                # ==== 获得观测值 ====
                ## [obs] dim=3
                omega = omega * ang_vel_scale

                ## [obs] dim=2
                roll, pitch, yaw = euler_from_quaternion(quat)
                imu_obs = np.array([roll, pitch], dtype=np.float32)  # [roll, pitch]

                ## [obs] dim=3
                delta_yaw, delta_next_yaw = 0, 0
                yaw_info = np.array([0, delta_yaw, delta_next_yaw], dtype=np.float32)

                ## [obs] dim=3
                cmd = np.array([0, 0, cmd[0]], dtype=np.float32)  # [0, 0, vx]

                mode = "parkour"
                ## [obs] dim=2
                if mode == "parkour":
                    parkour_walk = np.array([1, 0], dtype=np.float32) # parkour
                elif mode == "walk":
                    parkour_walk = np.array([0, 1], dtype=np.float32) # walk

                ## [obs] dim=12
                qj = (qj - default_angles) * dof_pos_scale

                ## [obs] dim=12
                dqj = dqj * dof_vel_scale

                ## [obs] dim=12
                last_action = action 

                ## [obs] contact force NOTE thinking rm contact_force
                # contact_force = np.array([d.cfrc_ext[0, 2], d.cfrc_ext[1, 2]], dtype=np.float32)
                # contact_filt = np.concatenate((d.sensordata["FL_foot_touch"] ,
                #                             d.sensordata["FR_foot_touch"] ,
                #                             d.sensordata["RL_foot_touch"] ,
                #                             d.sensordata["RR_foot_touch"] ))


                # 本体观测输入
                # import ipdb;ipdb.set_trace()
                obs_prop = np.concatenate((omega, imu_obs, yaw_info, cmd, parkour_walk, qj, dqj, last_action), axis=0)
                obs_prop_to_tensor = torch.from_numpy(obs_prop).unsqueeze(0).to(device).to(dtype=torch.float32)

                # history_obs_prop_to_tensor = torch.from_numpy(np.array(history_obs_prop)).to(device)
                # 深度图像和本体观测组合
                visual_update_interval = 5
                if counter % visual_update_interval == 0:
                    depth_image = _get_depth_image() ## 获得深度图像
                    depth_image_origin, depth_normalized, depth_normalized_to_tensor = _process_depth_image(depth_image) ## 处理深度图像
                    
                    ## last 初始化
                    if counter == 0:
                        last_depth_image = depth_normalized_to_tensor

                    ## depth encoder
                    # import ipdb;ipdb.set_trace()
                    depth_latent_yaw = depth_encoder(last_depth_image, obs_prop_to_tensor) # 融合
                    last_depth_image = depth_normalized_to_tensor

                    # import ipdb;ipdb.set_trace()

                # ==== 网络输入 ====
                obs_tensor = combine_all_obs(obs_prop_to_tensor, depth_latent_yaw, history_obs_prop, n_proprio, n_depth_latent, n_hist_len)


                # ==== 历史信息组合 ====
                # history_obs_prop.append(obs_prop.copy())
                history_obs_prop[: -1 ] = history_obs_prop[1:]
                history_obs_prop[-1:] = obs_prop.copy()


                # obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                # policy inference
                action = actor(obs_tensor).detach().cpu().numpy().squeeze()

                # transform action to target_dof_pos
                target_dof_pos = action * action_scale + default_angles


            
            counter += 1
            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()
            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


            # bgr = cv2.cvtColor(np.flipud(rgb), cv2.COLOR_RGB2BGR)
            # cv2.imshow('MuJoCo Camera Output', bgr)
            # cv2.imshow('Depth Map (Grayscale)', depth_image_origin)
            if cv2.waitKey(1) == 27:
                break



    # 退出OpenCV和MuJoCo
    glfw.terminate()
    del context



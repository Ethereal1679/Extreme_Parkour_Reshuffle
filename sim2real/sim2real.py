## only for dogs

from legged_gym import LEGGED_GYM_ROOT_DIR
from typing import Union
import numpy as np
import time
import torch
from torch import nn 

# 与通信有关
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
# from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
# from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
# from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.utils.crc import CRC


# 与控制有关
from common.command_helper import create_damping_cmd, create_zero_cmd, init_cmd_hg, init_cmd_go, MotorMode
from common.rotation_helper import get_gravity_orientation, transform_imu_data
from common.remote_controller import RemoteController, KeyMap

# config导入
from sim2real.sim2real_config import Config

# 与模型导入有关
from rsl_rl.modules.depth_backbone import RecurrentDepthBackbone, DepthOnlyFCBackbone58x87
import os



SIM2REAL = os.path.dirname(os.path.abspath(__file__))
print("SIM2REAL",SIM2REAL)



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



class Controller:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.remote_controller = RemoteController()

        ## ======== import models ==========
        self.device = "cuda"
        self.policy = torch.jit.load(config.policy_path, map_location=self.device)
        self.policy.eval()

        self.estimator = self.policy.estimator.estimator
        self.hist_encoder = self.policy.actor.history_encoder
        self.actor = self.policy.actor.actor_backbone


        vision_model = torch.load(self.config.vision_policy_path, map_location=self.device)
        depth_backbone = DepthOnlyFCBackbone58x87(None, 32, None)
        self.depth_encoder = RecurrentDepthBackbone(depth_backbone, None, eval_prop_n=self.config.n_proprio ).to(self.device)
        self.depth_encoder.load_state_dict(vision_model['depth_encoder_state_dict'])
        self.depth_encoder.to(self.device)
        self.depth_encoder.eval()


        ## ========= Initializing process variables ==========
        self.qj = np.zeros(config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(config.num_actions, dtype=np.float32)
        self.action = np.zeros(config.num_actions, dtype=np.float32)
        self.target_dof_pos = config.default_angles.copy()
        self.obs = np.zeros(config.num_obs, dtype=np.float32)
        self.cmd = np.array([0.0, 0, 0])
        self.counter = 0
        self.history_obs_prop = np.zeros((self.config.n_hist_len , self.config.n_proprio), dtype=np.float32)  


        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.low_state = unitree_go_msg_dds__LowState_()

        self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdGo)
        self.lowcmd_publisher_.Init()

        self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateGo)
        self.lowstate_subscriber.Init(self.LowStateGoHandler, 10)

        ### 获得相机数据 ###
        self.depth_camera_subscriber = ChannelSubscriber(config.depth_data_topic, None) 
        self.depth_camera_subscriber.Init(self.DepthDataHandler, 10)


        # wait for the subscriber to receive data
        self.wait_for_low_state()


        # Initialize the command msg
        # if config.msg_type == "hg":
        #     init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)
        # elif config.msg_type == "go":
        init_cmd_go(self.low_cmd, weak_motor=self.config.weak_motor)



    ## 处理深度相机数据
    def DepthDataHandler(self, msg):
        self.depth_data = torch.tensor(msg.data, dtype=torch.float32).reshape(1, 58, 87)  #.to(self.self.device)


    def LowStateGoHandler(self, msg: LowStateGo):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)


    # def send_cmd(self, cmd: Union[LowCmdGo, LowCmdHG]):
    #     cmd.crc = CRC().Crc(cmd)
    #     self.lowcmd_publisher_.Write(cmd)
    ## NOTE： rewrite
    def send_cmd(self, cmd: LowCmdGo):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")

    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        while self.remote_controller.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    # 移动到默认位置
    def move_to_default_pos(self):
        print("Moving to default pos.")
        # move time 2s
        total_time = 2
        num_step = int(total_time / self.config.control_dt)
        
        dof_idx = self.config.leg_joint2motor_idx #+ self.config.arm_waist_joint2motor_idx
        kps = self.config.kps #+ self.config.arm_waist_kps
        kds = self.config.kds #+ self.config.arm_waist_kds
        # default_pos = np.concatenate((self.config.default_angles, self.config.arm_waist_target), axis=0)
        default_pos = self.config.default_angles.copy() #+ self.config.arm_waist_target
        dof_size = len(dof_idx)
        
        # record the current pos
        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size):
            init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q
        
        # move to default pos
        for i in range(num_step):
            alpha = i / num_step
            for j in range(dof_size):
                motor_idx = dof_idx[j]
                target_pos = default_pos[j]
                self.low_cmd.motor_cmd[motor_idx].q = init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = kps[j]
                self.low_cmd.motor_cmd[motor_idx].kd = kds[j]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    # 默认位置状态
    def default_pos_state(self):
        print("Enter default pos state.")
        print("Waiting for the Button A signal...")
        while self.remote_controller.button[KeyMap.A] != 1:
            for i in range(len(self.config.leg_joint2motor_idx)):
                motor_idx = self.config.leg_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.default_angles[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0

            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)


    ## 组帧(深度图像+本体观测)
    def combine_all_obs(self, proprio, depth_latent_yaw, proprio_history, n_proprio, n_depth_latent, n_hist_len):
        depth_latent = depth_latent_yaw[:, :-2]
        yaw = depth_latent_yaw[:, -2:] * 1.5
        print('yaw: ', yaw)
        proprio[:, 6:8] = yaw
        lin_vel_latent = self.estimator(proprio)
        activation = nn.ELU()
        # import ipdb;ipdb.set_trace()
        # priv_latent = hist_encoder(activation, proprio_history.view(-1, n_hist_len, n_proprio))
        proprio_history = torch.from_numpy(proprio_history).to(self.device).unsqueeze(0)
        priv_latent = self.hist_encoder(activation, proprio_history)
        # import ipdb;ipdb.set_trace()
        obs = torch.cat([proprio, depth_latent, lin_vel_latent, priv_latent], dim=-1)
        return obs



    def run(self):
        self.counter += 1
        # Get the current joint position and velocity
        for i in range(len(self.config.leg_joint2motor_idx)):
            self.qj[i] = self.low_state.motor_state[self.config.leg_joint2motor_idx[i]].q
            self.dqj[i] = self.low_state.motor_state[self.config.leg_joint2motor_idx[i]].dq

        # imu_state quaternion: w, x, y, z
        quat = self.low_state.imu_state.quaternion
        ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)


        # create observation
        # gravity_orientation = get_gravity_orientation(quat)
        num_actions = self.config.num_actions
        action = np.zeros(num_actions, dtype=np.float32)



        self.cmd[0] = self.remote_controller.ly
        self.cmd[1] = self.remote_controller.lx * -1
        self.cmd[2] = self.remote_controller.rx * -1




        # ==== 获得观测值 ====
        ## [obs] dim=3
        omega = ang_vel * self.config.ang_vel_scale

        ## [obs] dim=2
        roll, pitch, yaw = euler_from_quaternion(quat)
        imu_obs = np.array([roll, pitch], dtype=np.float32)  # [roll, pitch]

        ## [obs] dim=3
        delta_yaw, delta_next_yaw = 0, 0
        yaw_info = np.array([0, delta_yaw, delta_next_yaw], dtype=np.float32)

        ## [obs] dim=3
        cmd = np.array([0, 0, self.cmd[0]], dtype=np.float32)  # [0, 0, vx]

        mode = "parkour"
        ## [obs] dim=2
        if mode == "parkour":
            parkour_walk = np.array([1, 0], dtype=np.float32) # parkour
        elif mode == "walk":
            parkour_walk = np.array([0, 1], dtype=np.float32) # walk

        ## [obs] dim=12
        qj_obs = self.qj.copy()
        qj_obs = (qj_obs - self.config.default_angles) * self.config.dof_pos_scale

        ## [obs] dim=12
        dqj_obs = self.dqj.copy()
        dqj_obs = dqj_obs * self.config.dof_vel_scale

        ## [obs] dim=12
        last_action = action 




        # 本体观测输入
        self.obs = np.concatenate((omega, imu_obs, yaw_info, cmd, parkour_walk, qj_obs, dqj_obs, last_action), axis=0)
        obs_tensor = torch.from_numpy(self.obs).unsqueeze(0)

        # 深度图像和本体观测组合
        visual_update_interval = 5
        if self.counter % visual_update_interval == 0:
            depth_image = self.depth_data               ## 获得深度图像
            # depth_image_origin, depth_normalized, depth_normalized_to_tensor = _process_depth_image(depth_image) ## 处理深度图像
            ## last 初始化
            if self.counter == 0:
                self.last_depth_image = depth_image
            ## depth encoder
            # import ipdb;ipdb.set_trace()
            depth_latent_yaw = self.depth_encoder(self.last_depth_image, obs_tensor) # 融合
            self.last_depth_image = depth_image


        # ==== 网络输入 ====
        obs_tensor = self.combine_all_obs(obs_tensor, depth_latent_yaw, self.history_obs_prop, self.config.n_proprio, self.config.n_depth_latent, self.config.n_hist_len)


        # ==== 历史信息组合 ====
        # history_obs_prop.append(obs_prop.copy())
        self.history_obs_prop[: -1 ] = self.history_obs_prop[1:]
        self.history_obs_prop[-1:] = self.obs_prop.copy()
        

        ### 输出action
        self.action = self.actor(obs_tensor).detach().numpy().squeeze()

        # transform action to target_dof_pos
        target_dof_pos = self.config.default_angles + self.action * self.config.action_scale

        # Build low cmd  电机指令下发
        for i in range(len(self.config.leg_joint2motor_idx)):
            motor_idx = self.config.leg_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = target_dof_pos[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0


        # send the command
        self.send_cmd(self.low_cmd)

        time.sleep(self.config.control_dt)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, help="network interface")
    parser.add_argument("config", type=str, help="config file name in the configs folder", default="g1.yaml")
    args = parser.parse_args()

    ################### self DIY #####################
    args.config = "go2.yaml"
    args.net = "enp0s3"

    # Load config
    config_path = f"{SIM2REAL}/configs/{args.config}"
    config = Config(config_path)

    # Initialize DDS communication
    ChannelFactoryInitialize(0, args.net)

    controller = Controller(config)

    # Enter the zero torque state, press the start key to continue executing
    controller.zero_torque_state()

    # Move to the default position
    controller.move_to_default_pos()

    # Enter the default position state, press the A key to continue executing
    controller.default_pos_state()


    while True:
        try:
            controller.run()
            # Press the select key to exit
            if controller.remote_controller.button[KeyMap.select] == 1:
                print("紧急中断")
                break
        except KeyboardInterrupt:
            break
    # Enter the damping state
    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
    print("Exit")





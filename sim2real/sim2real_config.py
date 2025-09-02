from legged_gym import LEGGED_GYM_ROOT_DIR
import numpy as np
import yaml
import os

SIM2REAL = os.path.dirname(os.path.abspath(__file__))
print("SIM2REAL",SIM2REAL)

class Config:
    def __init__(self, file_path) -> None:
        with open(file_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

            self.control_dt = config["control_dt"]

            # self.msg_type = config["msg_type"]
            # self.imu_type = config["imu_type"]

            self.weak_motor = []
            if "weak_motor" in config:
                self.weak_motor = config["weak_motor"]

            self.lowcmd_topic = config["lowcmd_topic"]
            self.lowstate_topic = config["lowstate_topic"]
            self.depth_data_topic = config["depth_data_topic"] ## add camera topic

            self.policy_path = config["policy_path"].replace("{SIM2REAL}", SIM2REAL)
            self.vision_policy_path = config["vision_policy_path"].replace("{SIM2REAL}", SIM2REAL) ## add vision policy

            self.leg_joint2motor_idx = config["leg_joint2motor_idx"]
            self.kps = config["kps"]
            self.kds = config["kds"]
            self.default_angles = np.array(config["default_angles"], dtype=np.float32)

            ## scales
            self.ang_vel_scale = config["ang_vel_scale"]
            self.dof_pos_scale = config["dof_pos_scale"]
            self.dof_vel_scale = config["dof_vel_scale"]
            self.action_scale = config["action_scale"]
            self.cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)


            self.max_cmd = np.array(config["max_cmd"], dtype=np.float32)

            self.num_actions = config["num_actions"]
            
            ### prop observation
            self.n_proprio = config["n_proprio"]
            self.n_hist_len = config["n_hist_len"]
            self.n_depth_latent = config["n_depth_latent"]
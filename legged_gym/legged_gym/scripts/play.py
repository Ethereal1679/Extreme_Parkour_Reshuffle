# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import code

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from isaacgym import gymtorch, gymapi, gymutil
import numpy as np
import torch
import torch.nn as nn
import cv2
from collections import deque
import statistics
import faulthandler
from copy import deepcopy
import matplotlib.pyplot as plt
from time import time, sleep
from legged_gym.utils import webviewer
from rsl_rl.modules.actor_critic import Actor, StateHistoryEncoder, get_activation, ActorCriticRMA
from rsl_rl.modules.estimator import Estimator
from rsl_rl.modules.depth_backbone import DepthOnlyFCBackbone58x87, RecurrentDepthBackbone


### load path
def get_load_path(root, load_run=-1, checkpoint=-1, model_name_include="model"):
    if not os.path.isdir(root):  # use first 4 chars to mactch the run name
        model_name_cand = os.path.basename(root)
        model_parent = os.path.dirname(root)
        model_names = os.listdir(model_parent)
        model_names = [name for name in model_names if os.path.isdir(os.path.join(model_parent, name))]
        for name in model_names:
            if len(name) >= 6:
                if name[:6] == model_name_cand:
                    root = os.path.join(model_parent, name)
    if checkpoint==-1:
        models = [file for file in os.listdir(root) if model_name_include in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
        checkpoint = model.split("_")[-1].split(".")[0]
    else:
        model = "model_{}.pt".format(checkpoint) 

    load_path = os.path.join(root, model)
    return load_path, checkpoint



## import networks
class HardwareVisionNN(nn.Module):
    def __init__(self,  num_prop,
                        num_scan,
                        num_priv_latent, 
                        num_priv_explicit,
                        num_hist,
                        num_actions,
                        tanh,
                        actor_hidden_dims=[512, 256, 128],
                        scan_encoder_dims=[128, 64, 32],
                        depth_encoder_hidden_dim=512,
                        activation='elu',
                        priv_encoder_dims=[64, 20]
                        ):
        super(HardwareVisionNN, self).__init__()

        self.num_prop = num_prop
        self.num_scan = num_scan
        self.num_hist = num_hist
        self.num_actions = num_actions
        self.num_priv_latent = num_priv_latent
        self.num_priv_explicit = num_priv_explicit
        num_obs = num_prop + num_scan + num_hist*num_prop + num_priv_latent + num_priv_explicit
        self.num_obs = num_obs
        activation = get_activation(activation)

        self.actor = Actor(num_prop, num_scan, num_actions, scan_encoder_dims, actor_hidden_dims, priv_encoder_dims, num_priv_latent, num_priv_explicit, num_hist, activation, tanh_encoder_output=tanh)

        self.estimator = Estimator(input_dim=num_prop, output_dim=num_priv_explicit, hidden_dims=[128, 64])
        
    def forward(self, obs, depth_latent):
        obs[:, self.num_prop+self.num_scan : self.num_prop+self.num_scan+self.num_priv_explicit] = self.estimator(obs[:, :self.num_prop])
        return self.actor(obs, hist_encoding=True, eval=False, scandots_latent=depth_latent)





def get_load_path(root, load_run=-1, checkpoint=-1, model_name_include="model"):
    if checkpoint==-1:
        models = [file for file in os.listdir(root) if model_name_include in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
        checkpoint = model.split("_")[-1].split(".")[0]
    return model, checkpoint


def play(args):
    if args.web:
        web_viewer = webviewer.WebViewer()
    faulthandler.enable()
    exptid = args.exptid
    log_pth = "../../logs/{}/".format(args.proj_name) + args.exptid

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    if args.nodelay:
        env_cfg.domain_rand.action_delay_view = 0
    env_cfg.env.num_envs = 16 if not args.save else 64
    env_cfg.env.episode_length_s = 60
    env_cfg.commands.resampling_time = 60
    ### 
    env_cfg.terrain.num_rows = 3#5
    env_cfg.terrain.num_cols = 3#5
    env_cfg.terrain.height = [0.02, 0.02]
    env_cfg.terrain.terrain_dict = {"smooth slope": 0., 
                                    "rough slope up": 0.0,
                                    "rough slope down": 0.0,
                                    "rough stairs up": 0., 
                                    "rough stairs down": 0., 
                                    "discrete": 0., 
                                    "stepping stones": 0.0,
                                    "gaps": 0., 
                                    "smooth flat": 0,
                                    "pit": 0.0,
                                    "wall": 0.0,
                                    "platform": 0.,
                                    "large stairs up": 0.,
                                    "large stairs down": 0.,
                                    "parkour": 0.2,
                                    "parkour_hurdle": 0.2,
                                    "parkour_flat": 0.,
                                    "parkour_step": 0.2,
                                    "parkour_gap": 0.2, 
                                    "demo": 0.2}
    
    env_cfg.terrain.terrain_proportions = list(env_cfg.terrain.terrain_dict.values())
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_difficulty = True
    
    env_cfg.depth.angle = [0, 1]
    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_interval_s = 6
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False

    env_cfg.env.debug_show = args.debug_show
    
    depth_latent_buffer = []
    # prepare environment
    env: LeggedRobot
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    if args.web:
        web_viewer.setup(env)

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg, log_pth = task_registry.make_alg_runner(log_root = log_pth, env=env, name=args.task, args=args, train_cfg=train_cfg, return_log_dir=True)


    ## add export logits
    if EXPORT_POLICY:
        load_run = "../../logs/train_go2/" + args.exptid
        checkpoint = args.checkpoint
        device = torch.device('cpu')
        n_priv_explicit = env_cfg.env.n_priv
        n_priv_latent = env_cfg.env.n_priv_latent
        num_scan = env_cfg.env.n_scan
        num_actions = env_cfg.env.num_actions
        n_proprio = env_cfg.env.n_proprio
        history_len = env_cfg.env.history_len
        policy = HardwareVisionNN(n_proprio, num_scan, n_priv_latent, n_priv_explicit, history_len, num_actions, tanh=False).to(device)
        load_path, checkpoint = get_load_path(root=load_run, checkpoint=-1)
        path = os.path.join(load_run, load_path)

        print(f"Loading model from: {path}")
        ac_state_dict = torch.load(path, map_location=device)

        policy.actor.load_state_dict(ac_state_dict['depth_actor_state_dict'], strict=True)
        policy.estimator.load_state_dict(ac_state_dict['estimator_state_dict'])


        policy = policy.to(device)#.cpu()
        if not os.path.exists(os.path.join(load_run, "traced")):
            os.mkdir(os.path.join(load_run, "traced"))
        state_dict = {'depth_encoder_state_dict': ac_state_dict['depth_encoder_state_dict']}
        torch.save(state_dict, os.path.join(load_run, "traced", args.exptid + "-" + str(checkpoint) + "-vision_weight.pt"))

        # Save the traced actor
        policy.eval()
        with torch.no_grad(): 
            num_envs = 1
            
            obs_input = torch.ones(num_envs, n_proprio + num_scan + n_priv_explicit + n_priv_latent + history_len*n_proprio, device=device)
            depth_latent = torch.ones(1, 32, device=device)
            test = policy(obs_input, depth_latent)
            
            traced_policy = torch.jit.trace(policy, (obs_input, depth_latent))
            
            # traced_policy = torch.jit.script(policy)
            save_path = os.path.join(load_run, "traced", args.exptid + "-" + str(checkpoint) + "-base_jit.pt")
            traced_policy.save(save_path)
            print("Saved traced_actor at ", os.path.abspath(save_path))


    if args.use_jit:
        path = os.path.join(log_pth, "traced")
        model, checkpoint = get_load_path(root=path, checkpoint=args.checkpoint)
        path = os.path.join(path, model)
        print("Loading jit for policy: ", path)
        policy_jit = torch.jit.load(path, map_location=env.device)
    else:
        policy = ppo_runner.get_inference_policy(device=env.device)
    estimator = ppo_runner.get_estimator_inference_policy(device=env.device)
    if env.cfg.depth.use_camera:
        depth_encoder = ppo_runner.get_depth_encoder_inference_policy(device=env.device)

    actions = torch.zeros(env.num_envs, 12, device=env.device, requires_grad=False)
    infos = {}
    infos["depth"] = env.depth_buffer.clone().to(ppo_runner.device)[:, -1] if ppo_runner.if_depth else None

    # for i in range(10*int(env.max_episode_length)):
    state_lange_time = 2000
    for i in range(state_lange_time):
        if args.use_jit:
            if env.cfg.depth.use_camera:
                if infos["depth"] is not None:
                    depth_latent = torch.ones((env_cfg.env.num_envs, 32), device=env.device)
                    actions, depth_latent = policy_jit(obs.detach(), True, infos["depth"], depth_latent)
                else:
                    depth_buffer = torch.ones((env_cfg.env.num_envs, 58, 87), device=env.device)
                    actions, depth_latent = policy_jit(obs.detach(), False, depth_buffer, depth_latent)
            else:
                obs_jit = torch.cat((obs.detach()[:, :env_cfg.env.n_proprio+env_cfg.env.n_priv], obs.detach()[:, -env_cfg.env.history_len*env_cfg.env.n_proprio:]), dim=1)
                actions = policy(obs_jit)
        else:
            if env.cfg.depth.use_camera:
                if infos["depth"] is not None:
                    obs_student = obs[:, :env.cfg.env.n_proprio].clone()
                    obs_student[:, 6:8] = 0
                    depth_latent_and_yaw = depth_encoder(infos["depth"], obs_student)
                    depth_latent = depth_latent_and_yaw[:, :-2]
                    yaw = depth_latent_and_yaw[:, -2:]
                obs[:, 6:8] = 1.5*yaw
                    
            else:
                depth_latent = None
            
            if hasattr(ppo_runner.alg, "depth_actor"):
                actions = ppo_runner.alg.depth_actor(obs.detach(), hist_encoding=True, scandots_latent=depth_latent)
            else:
                actions = policy(obs.detach(), hist_encoding=True, scandots_latent=depth_latent)
            
        obs, _, rews, dones, infos = env.step(actions.detach())
        if args.web:
            web_viewer.render(fetch_results=True,
                        step_graphics=True,
                        render_all_camera_sensors=True,
                        wait_for_page_load=True)
        print("time:", env.episode_length_buf[env.lookat_id].item() / 50, 
              "cmd vx", env.commands[env.lookat_id, 0].item(),
              "actual vx", env.base_lin_vel[env.lookat_id, 0].item(), )
        
        id = env.lookat_id
        



if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    args.debug_show = True

    NAME="S"

    ## teacher
    if NAME == "T":
        args.task = "go2"  # Set the task to A1
        args.proj_name = "train_go2"
        args.exptid  = "20250811_014015_Teacher_go2_original"
        args.num_envs = 50


    ## student
    if NAME == "S":
        args.task = "go2"  # Set the task to A1
        args.proj_name = "train_go2"
        args.exptid  = "20250813_162402_Student_go2_original"
        args.num_envs = 20
        args.delay = True
        args.use_camera = True
        args.checkpoint = 385000


        # args.task = "go1"  # Set the task to A1
        # args.proj_name = "train"
        # args.exptid  = "Student-test1"
        # args.num_envs = 15
        # args.delay = True
        # args.use_camera = True

    play(args)



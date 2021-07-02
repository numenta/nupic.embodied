# ------------------------------------------------------------------------------
#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2021, Numenta, Inc.  Unless you have an agreement
#  with Numenta, Inc., for a separate license for this software code, the
#  following terms and conditions apply:
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero Public License version 3 as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Affero Public License for more details.
#
#  You should have received a copy of the GNU Affero Public License
#  along with this program.  If not, see http://www.gnu.org/licenses.
#
#  http://numenta.org/licenses/
#
# ------------------------------------------------------------------------------

# Example Usage:
# python enjoy.py --exp_name=Name --num_timesteps=1024

if __name__ == "__main__":
    import argparse

    # from real_robots.envs import REALRobotEnv
    # from envs.wrappers import CartesianControlDiscrete
    from nupic.embodied.policies.curious_cnn_policy import CnnPolicy
    import torch
    import numpy as np
    from learn import make_env_all_params

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Experiment Parameters:
    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument("--expID", type=str, default="000")
    parser.add_argument("--seed", help="RNG seed", type=int, default=0)
    parser.add_argument("--download_model_from", type=str, default="")

    # Environment parameters:
    parser.add_argument(
        "--env", help="environment ID", default="BreakoutNoFrameskip-v4", type=str
    )
    parser.add_argument("--num_timesteps", type=int, default=int(1024))
    parser.add_argument(
        "--max-episode-steps",
        help="maximum number of timesteps for episode",
        default=4500,
        type=int,
    )
    parser.add_argument("--env_kind", type=str, default="atari")
    parser.add_argument("--noop_max", type=int, default=30)
    parser.add_argument("--act_repeat", type=int, default=10)
    parser.add_argument("--stickyAtari", action="store_true", default=True)
    parser.add_argument("--crop_obs", action="store_true", default=True)
    parser.add_argument("--touch_reward", action="store_true", default=False)
    parser.add_argument("--random_force", action="store_true", default=False)

    args = parser.parse_args()

    try:
        ob_mean = np.load("./statistics/" + args.env + "/ob_mean.npy")
        ob_std = np.load("./statistics/" + args.env + "/ob_std.npy")
        print("loaded environment statistics")
    except FileNotFoundError:
        print("No statistics file found. Run training to create one.")

    """if args.env_kind == "roboarm":
        env = REALRobotEnv(objects=3, action_type="cartesian")
        env = CartesianControlDiscrete(
            env,
            crop_obs=args.crop_obs,
            repeat=args.act_repeat,
            touch_reward=args.touch_reward,
        )
        env.render("human")"""

    env = make_env_all_params(0, args.__dict__)

    if torch.cuda.is_available():
        print("GPU detected")
        dev_name = "cuda:0"
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        print("no GPU detected, using CPU instead.")
        dev_name = "cpu"
    device = torch.device(dev_name)
    print("device: " + str(device))

    policy = CnnPolicy(
        scope="policy",
        device=device,
        ob_space=env.observation_space,
        ac_space=env.action_space,
        hidden_dim=512,
        feature_dim=512,
        ob_mean=ob_mean,
        ob_std=ob_std,
        layernormalize=False,
        nonlinear=torch.nn.LeakyReLU,
    )

    if args.download_model_from != "":
        import wandb

        # TODO: This creates a new run. Find a way to download artifact without starting
        # to log to wandb
        run = wandb.init(
            project="embodiedAI",
            name=args.exp_name,
        )
        artifact = run.use_artifact(args.download_model_from, type="model")
        model_path = artifact.download()
    else:
        model_path = "./models/" + args.exp_name
    # Load the trained policy from ./models/exp_name/model.pt
    print("Loading model from " + str(model_path + "/model.pt"))
    checkpoint = torch.load(model_path + "/model.pt")

    policy.features_model.load_state_dict(checkpoint["policy_features"])
    policy.pd_hidden.load_state_dict(checkpoint["policy_hidden"])
    policy.pd_head.load_state_dict(checkpoint["policy_pd_head"])
    policy.vf_head.load_state_dict(checkpoint["policy_vf_head"])

    observation = env.reset()
    observation = np.array(observation)

    reward, done = 0, False
    last_done = 0
    for s in range(args.num_timesteps):
        # Quick fix right now for shape error concatenate twice the same obs
        acs, vpreds, nlps = policy.get_ac_value_nlp(
            np.array([[observation, observation]])
        )
        if args.env_kind == "roboarm":
            print(env.act_dict[acs[0]])
        # self.env_step(l, acs)

        # action = env.action_space.sample()
        env.render()
        observation, reward, done, info = env.step(acs[0])
        if done:
            print("episode length: " + str(s - last_done))
            last_done = s
            observation = env.reset()
        observation = np.array(observation)
    env.close()

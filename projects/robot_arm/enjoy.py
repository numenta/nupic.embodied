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
    parser.add_argument("--dyn_from_pixels", type=int, default=0)
    parser.add_argument("--use_done", type=int, default=0)
    parser.add_argument("--ext_coeff", type=float, default=0.0)
    parser.add_argument("--int_coeff", type=float, default=1.0)
    parser.add_argument("--layernorm", type=int, default=0)
    parser.add_argument(
        "--feat_learning",
        type=str,
        default="none",
        choices=["none", "idf", "vaesph", "vaenonsph"],
    )
    parser.add_argument("--num_dynamics", type=int, default=5)
    parser.add_argument("--dont_use_disagreement", action="store_false", default=True)
    parser.add_argument("--load", action="store_true", default=True)

    # Environment parameters:
    parser.add_argument(
        "--env", help="environment ID", default="BreakoutNoFrameskip-v4", type=str
    )
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

    policy = CnnPolicy(
        scope="policy",
        ob_space=env.observation_space,
        ac_space=env.action_space,
        hidden_dim=512,
        feature_dim=512,
        ob_mean=ob_mean,
        ob_std=ob_std,
        layernormalize=False,
        nonlinear=torch.nn.LeakyReLU,
    )

    # TODO: Add pytorch model loading.

    observation = env.reset()
    observation = np.array(observation)

    reward, done = 0, False
    for action in range(50):
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
        observation = np.array(observation)
    env.close()

from cliff_walking_qlearning import *
parser = argparse.ArgumentParser()
parser.add_argument('--setting', type=int, default=0)
parser.add_argument('--debug', type=int, default=0)
args = parser.parse_args()

setting = args.setting
debug = args.debug

date = '2021-10-05'

# env1: time_in_state=True, wind_only_affect_right_action = False,
# env2: time_in_state=False, wind_only_affect_right_action = True,

time_limit = 50
n_episode = int(3e5)

if args.debug == 1:
    n_episode = 100

n_seed = 5
world_size_list = [
[4, 3],
[8, 6],
[12, 9],
[16, 12],
]

wind_prob_list = [
    0, 0.1, 0.3, 0.5
                  ]

gamma_list = [0.9, 0.99, 0.999, 1]
# gamma_list = [1]
alpha_list = [0.1, 0.01, 0.001]
# alpha_list = [0.1]

log_interval = 100
uniform_start = True

i_setting = -1
get_optimal_value_for_each_setting = True

for world_size in world_size_list:
    world_width, world_height = world_size
    for wind_prob in wind_prob_list:
        for main_env_setting in (0, 1):
            if main_env_setting == 0:
                time_in_state = False
                wind_only_affect_right_action = True
            else:
                time_in_state = True
                wind_only_affect_right_action = False

            # first obtain optimal values
            if not get_optimal_value_for_each_setting:
                _, optimal_q_val_map, _ = value_iteration(world_width, world_height, wind_prob, time_limit,
                                                      wind_only_affect_right_action, time_in_state)
            for multi_update in (True, False):
                for gamma in gamma_list:
                    for alpha in alpha_list:
                        for seed in range(n_seed):
                            i_setting += 1
                            if i_setting != setting:
                                continue

                            if get_optimal_value_for_each_setting:
                                _, optimal_q_val_map, _ = value_iteration(world_width, world_height, wind_prob, time_limit,
                                                                          wind_only_affect_right_action, time_in_state)

                            datafile_name_suffix = 'QL_N%d_W%d_H%d_TL%d_WP%s_TinS%s_M%s_A%s_G%s_seed%d_%s.csv' % (
                             n_episode, world_width, world_height, time_limit,
                            str(wind_prob), time_in_state, multi_update, str(alpha), str(gamma), seed, date)

                            print("suffix:", datafile_name_suffix)

                            # when we have all the settings, we get the q diff to optimal q values and the performance score values
                            avg_diff, score, q_update_diff = PerformQLearning(world_width, world_height, wind_prob, time_limit,
                                                n_episode, wind_only_affect_right_action, time_in_state,
                                                multi_update, uniform_start, seed, optimal_q_val_map,
                                                visualize=False, log_interval=log_interval, eval_interval=log_interval, n_eval=10,
                                                                              gamma=gamma, alpha=alpha)

                            avg_diff, score, q_update_diff = np.array(avg_diff), np.array(score), np.array(q_update_diff)

                            data_to_save_list = [avg_diff, score, q_update_diff]
                            datafile_name_prefix_list = ['QOptDiff', 'Score', 'QUpDiff']

                            for z in range(3):
                                save_path = datafile_name_prefix_list[z] + datafile_name_suffix
                                save_path = os.path.join('../data', save_path)
                                data_to_save_list[z].tofile(save_path, sep=',')

print(i_setting)
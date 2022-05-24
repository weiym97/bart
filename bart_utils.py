







def generate_option(const_reward_all, const_subexplode_prob_all, penalty_all, model_name_all, idx):
    total_num = len(const_reward_all) * len(const_subexplode_prob_all) * len(penalty_all) * len(model_name_all)

    total_num //= len(const_reward_all)
    const_reward = const_reward_all[idx // total_num]
    idx = idx % total_num

    total_num //= len(const_subexplode_prob_all)
    const_subexplode_prob = const_subexplode_prob_all[idx // total_num]
    idx = idx % total_num

    total_num //= len(penalty_all)
    penalty = penalty_all[idx // total_num]
    idx = idx % total_num

    total_num //= len(model_name_all)
    model_name = model_name_all[idx // total_num]

    trial_id = 'const_reward_' + str(const_reward) + '_const_subexplode_prob_' + str(const_subexplode_prob) \
               + '_penalty_' + str(penalty) + '_model_name_' + str(model_name)

    return trial_id, const_reward, const_subexplode_prob, penalty, model_name
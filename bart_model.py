import sys
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class FourparamBart():
    def __init__(self, max_pump, explode_prob, accu_reward, num_trial=50):
        self.max_pump = max_pump
        self.explode_prob = explode_prob
        self.accu_reward = accu_reward
        self.reward = np.diff(accu_reward)
        self.num_trial = num_trial

    def generate_data(self, phi, eta, gamma, tau):
        pumps = np.zeros(self.num_trial)
        explode = np.zeros(self.num_trial)
        # Record total pumps and successful pumps
        n_success = 0
        n_pumps = 0
        for i in range(self.num_trial):
            # Calculate the optimal number of pumps
            p_burst = 1 - (phi + eta * n_success) / (1 + eta * n_pumps)
            optimal_pump = -gamma / np.log(1 - p_burst)

            ### probability of burst
            burst = (self.explode_prob > np.random.uniform())
            for j in range(self.max_pump):
                # Subject probability of pump
                p_pump = 1 / (1 + np.exp(tau * (j + 1 - optimal_pump)))
                pump = int(np.random.binomial(1, p_pump, 1)) > 0
                if not pump:
                    pumps[i] = j
                    explode[i] = 0
                    break
                elif burst[j]:
                    pumps[i] = j + 1
                    explode[i] = 1
                    break
            n_success += pumps[i] - explode[i]
            n_pumps += pumps[i]
        return pumps.astype(np.int32), explode.astype(np.int32)


class EWBart():
    def __init__(self, max_pump, explode_prob, accu_reward, num_trial=50, const_subexplode_prob=True):
        self.max_pump = max_pump
        self.explode_prob = explode_prob
        self.accu_reward = accu_reward
        self.reward = np.diff(accu_reward)
        self.num_trial = num_trial
        self.const_subexplode_prob = const_subexplode_prob

    def generate_data(self, psi, xi, rho, Lambda, tau):
        pumps = np.zeros(self.num_trial)
        explode = np.zeros(self.num_trial)
        if self.const_subexplode_prob:
            n_success = 0
            n_pumps = 0
        else:
            n_success = np.zeros(self.max_pump)
            n_pumps = np.zeros(self.max_pump)
        for i in range(self.num_trial):
            burst = (self.explode_prob > np.random.uniform())
            for j in range(self.max_pump):
                if self.const_subexplode_prob:
                    n_pumps_use = n_pumps
                    n_success_use = n_success
                else:
                    n_pumps_use = n_pumps[j]
                    n_success_use = n_success[j]
                if n_pumps_use > 0:
                    p_temp = (n_pumps_use - n_success_use) / n_pumps_use
                else:
                    p_temp = 0
                p_burst = np.exp(-xi * n_pumps_use) * psi + (1 - np.exp(-xi * n_pumps_use)) * p_temp
                utility = (1 - p_burst) * self.reward[j] ** rho - p_burst * Lambda * (
                        2 * self.accu_reward[j]) ** rho
                p_pump = 1 / (1 + np.exp(-tau * utility))
                pump = int(np.random.binomial(1, p_pump, 1)) > 0
                if not pump:
                    pumps[i] = j
                    explode[i] = 0
                    break
                elif burst[j]:
                    pumps[i] = j + 1
                    explode[i] = 1
                    break
            if self.const_subexplode_prob:
                n_success += pumps[i] - explode[i]
                n_pumps += pumps[i]
            else:
                n_success[:int(pumps[i] - explode[i])] += 1
                n_pumps[:int(pumps[i])] += 1
        return pumps.astype(np.int32), explode.astype(np.int32)


class EWMVBart():
    def __init__(self, max_pump, explode_prob, accu_reward, num_trial=50, const_subexplode_prob=True):
        self.max_pump = max_pump
        self.explode_prob = explode_prob
        self.accu_reward = accu_reward
        self.reward = np.diff(accu_reward)
        self.num_trial = num_trial
        self.const_subexplode_prob = const_subexplode_prob

    def generate_data(self, psi, xi, rho, Lambda, tau):
        pumps = np.zeros(self.num_trial)
        explode = np.zeros(self.num_trial)
        if self.const_subexplode_prob:
            n_success = 0
            n_pumps = 0
        else:
            n_success = np.zeros(self.max_pump)
            n_pumps = np.zeros(self.max_pump)
        for i in range(self.num_trial):
            burst = (self.explode_prob > np.random.uniform())
            for j in range(self.max_pump):
                if self.const_subexplode_prob:
                    n_pumps_use = n_pumps
                    n_success_use = n_success
                else:
                    n_pumps_use = n_pumps[j]
                    n_success_use = n_success[j]
                if n_pumps_use > 0:
                    p_temp = (n_pumps_use - n_success_use) / n_pumps_use
                else:
                    p_temp = 0
                p_burst = np.exp(-xi * n_pumps_use) * psi + (1 - np.exp(-xi * n_pumps_use)) * p_temp
                utility = (1 - p_burst) * self.reward[j] - 2 * p_burst * Lambda * self.accu_reward[j] + \
                          rho * p_burst * (1 - p_burst) * (self.reward[j] + 2 * Lambda * self.accu_reward[j])
                p_pump = 1 / (1 + np.exp(-tau * utility))
                pump = int(np.random.binomial(1, p_pump, 1)) > 0
                if not pump:
                    pumps[i] = j
                    explode[i] = 0
                    break
                elif burst[j]:
                    pumps[i] = j + 1
                    explode[i] = 1
                    break
            if self.const_subexplode_prob:
                n_success += pumps[i] - explode[i]
                n_pumps += pumps[i]
            else:
                n_success[:int(pumps[i] - explode[i])] += 1
                n_pumps[:int(pumps[i])] += 1
        return pumps.astype(np.int32), explode.astype(np.int32)


class NEWBart():
    def __init__(self, max_pump, explode_prob, accu_reward, num_trial=50):
        self.max_pump = max_pump
        self.explode_prob = explode_prob
        self.accu_reward = accu_reward
        self.reward = np.diff(accu_reward)
        self.num_trial = num_trial

    def generate_data(self, Q_0, alpha, gamma, Lambda, tau):
        pumps = np.zeros(self.num_trial)
        explode = np.zeros(self.num_trial)
        # Q represent the subjective optimal pump reward
        Q = Q_0
        # C represent the confidence
        C = 0
        # L represent the accumulative loss aversion
        L = 0

        for i in range(self.num_trial):
            burst = (self.explode_prob > np.random.uniform())
            for j in range(self.max_pump):
                # Subject probability of pump
                p_pump = 1 / (1 + np.exp(tau * (self.accu_reward[j] - (Q + gamma * C - Lambda * L))))
                pump = int(np.random.binomial(1, p_pump, 1)) > 0
                if not pump:
                    pumps[i] = j
                    explode[i] = 0
                    Q = Q + alpha * max(self.accu_reward[j] - Q, 0)
                    C = C + alpha * (1 - C)
                    L = L + alpha * (self.accu_reward[j] - L)
                    break
                elif burst[j]:
                    pumps[i] = j + 1
                    explode[i] = 1
                    Q = Q - alpha * self.accu_reward[j]
                    C = (1 - alpha) * C
                    L = (1 - alpha) * L
                    break

        return pumps.astype(np.int32), explode.astype(np.int32)


class RLBart():
    def __init__(self, max_pump, explode_prob, accu_reward, num_trial=50, model_type='1'):
        self.max_pump = max_pump
        self.explode_prob = explode_prob
        self.accu_reward = accu_reward
        self.reward = np.diff(accu_reward)
        self.num_trial = num_trial
        # We have two model types (for details please turn to yuque)
        self.model_type = model_type

    def generate_data(self, Q_0, alpha, gamma, Lambda, tau, return_Q=False):
        pumps = np.zeros(self.num_trial)
        explode = np.zeros(self.num_trial)
        # Initialise subjective optimal pump reward
        Q = Q_0
        if return_Q:
            Q_history = np.zeros(self.num_trial)
        for i in range(self.num_trial):
            burst = (self.explode_prob > np.random.uniform())
            for j in range(self.max_pump):
                # Subject probability of pump
                p_pump = 1 / (1 + np.exp(tau * (self.accu_reward[j] - Q)))
                pump = int(np.random.binomial(1, p_pump, 1)) > 0
                if not pump:
                    pumps[i] = j
                    explode[i] = 0
                    Q = self.accu_reward[j] - alpha * (self.accu_reward[j] - Q) + gamma
                    if return_Q:
                        Q_history[i] = Q
                    break
                elif burst[j]:
                    pumps[i] = j + 1
                    explode[i] = 1
                    if self.model_type == '1':
                        Q = Q - Lambda * self.accu_reward[j]
                    elif self.model_type == '2':
                        Q = Q + Lambda * min(self.accu_reward[j] - Q, 0)
                    else:
                        raise ValueError('Invalid model type!')
                    if return_Q:
                        Q_history[i] = Q
                    break
        if return_Q:
            return pumps.astype(np.int32), explode.astype(np.int32), Q_history
        else:
            return pumps.astype(np.int32), explode.astype(np.int32)

    #def compute_likelihood(self,Q_0, alpha, gamma, Lambda, tau,pumps,explode):


class STLBart():
    def __init__(self, max_pump, explode_prob, accu_reward, num_trial=50):
        self.max_pump = max_pump
        self.explode_prob = explode_prob
        self.accu_reward = accu_reward
        self.reward = np.diff(accu_reward)
        self.num_trial = num_trial

    def generate_data(self, omega_0, vwin, vloss, tau, return_omega=False):
        pumps = np.zeros(self.num_trial)
        explode = np.zeros(self.num_trial)
        # Initialise subjective optimal pump reward
        omega = omega_0
        if return_omega:
            omega_history = np.zeros(self.num_trial)
        for i in range(self.num_trial):
            burst = (self.explode_prob > np.random.uniform())
            for j in range(self.max_pump):
                # Subject probability of pump
                p_pump = 1 / (1 + np.exp(tau * (j + 1 - omega * self.max_pump)))
                pump = int(np.random.binomial(1, p_pump, 1)) > 0
                if not pump:
                    pumps[i] = j
                    explode[i] = 0
                    omega = omega * (1 + vwin * j / self.max_pump)
                    if return_omega:
                        omega_history[i] = omega
                    break
                elif burst[j]:
                    pumps[i] = j + 1
                    explode[i] = 1
                    omega = omega * (1 - vloss * (1 - (j + 1) / self.max_pump))
                    if return_omega:
                        omega_history[i] = omega
                    break
        if return_omega:
            return pumps.astype(np.int32), explode.astype(np.int32), omega_history
        else:
            return pumps.astype(np.int32), explode.astype(np.int32)


class STLDBart():
    def __init__(self, max_pump, explode_prob, accu_reward, num_trial=50):
        self.max_pump = max_pump
        self.explode_prob = explode_prob
        self.accu_reward = accu_reward
        self.reward = np.diff(accu_reward)
        self.num_trial = num_trial

    def generate_data(self, omega_0, vwin, vloss, alpha, tau, return_omega=False):
        pumps = np.zeros(self.num_trial)
        explode = np.zeros(self.num_trial)
        # Initialise subjective optimal pump reward
        omega = omega_0
        if return_omega:
            omega_history = np.zeros(self.num_trial)
        for i in range(self.num_trial):
            burst = (self.explode_prob > np.random.uniform())
            for j in range(self.max_pump):
                # Subject probability of pump
                p_pump = 1 / (1 + np.exp(tau * (j + 1 - omega * self.max_pump)))
                pump = int(np.random.binomial(1, p_pump, 1)) > 0
                if not pump:
                    pumps[i] = j
                    explode[i] = 0
                    omega = omega * (1 + (vwin * j / self.max_pump) / (1 + alpha * i))
                    if return_omega:
                        omega_history[i] = omega
                    break
                elif burst[j]:
                    pumps[i] = j + 1
                    explode[i] = 1
                    omega = omega * (1 - (vloss * (1 - (j + 1) / self.max_pump)) / (1 + alpha * i))
                    if return_omega:
                        omega_history[i] = omega
                    break
        if return_omega:
            return pumps.astype(np.int32), explode.astype(np.int32), omega_history
        else:
            return pumps.astype(np.int32), explode.astype(np.int32)


def bart_generate_basic_info(config, save_dir):
    save_dir = save_dir + config['trial_id'] + '/'
    result = pd.read_excel(save_dir + 'result.xlsx')
    result.drop(['trial', 'pumps', 'explosion'], axis=1, inplace=True)
    result.drop_duplicates(inplace=True)
    result.to_excel(save_dir + 'basic_info.xlsx', index=False)


def bart_plot_stop_main(config, save_dir):
    save_dir = save_dir + config['trial_id'] + '/'
    result = pd.read_excel(save_dir + 'result.xlsx')
    plot_dir = save_dir + 'plot_stop/'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    max_subjID = int(max(result['SubjID']))

    for subjID in range(1, max_subjID + 1):
        subresult = result[result['SubjID'] == subjID]
        subresult = subresult[subresult['explosion'] == 0]
        plt.hist(subresult['pumps'], range=(0, config['max_pump'] - 1))
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=20)
        plt.xlabel('Pump Number', fontsize=20)
        plt.title('SubjectID ' + str(subjID), fontsize=25)
        plt.savefig(plot_dir + 'subjID=' + str(subjID) + '.jpg')
        plt.close()


def bart_plot_pump_prob(config, save_dir):
    save_dir = save_dir + config['trial_id'] + '/'
    result = pd.read_excel(save_dir + 'result.xlsx')
    plot_dir = save_dir + 'plot_pump_prob/'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    max_subjID = int(max(result['SubjID']))

    for subjID in range(1, max_subjID + 1):
        subresult = result[result['SubjID'] == subjID]
        subresult_succeed = subresult[subresult['explosion'] == 0]
        subresult_fail = subresult[subresult['explosion'] == 1]
        pumps_succeed = np.array(subresult_succeed['pumps'])
        pumps_fail = np.array(subresult_fail['pumps'])

        pumps = np.zeros(config['max_pump'] - 1)
        no_pumps = np.zeros(config['max_pump'] - 1)
        for i in range(config['max_pump'] - 1):
            no_pumps[i] = np.sum(pumps_succeed == i + 1)
            pumps[i] = np.sum(pumps_succeed > i + 1) + np.sum(pumps_fail == i + 1)
        pump_prob = pumps / (pumps + no_pumps + 1e-8)
        plt.plot(np.arange(1, config['max_pump']), pump_prob)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=20)
        plt.xlabel('Pump Number', fontsize=20)
        plt.title('SubjectID ' + str(subjID) + ' : pump probability', fontsize=25)

        plt.savefig(plot_dir + 'subjID=' + str(subjID) + '.jpg')
        plt.close()


if __name__ == '__main__':
    accu_reward = np.array([0.0, 0.0, 0.05, 0.15, 0.25, 0.55, 0.95, 1.45, 2.05, 2.75, 3.45, 4.25, 5.15, 6.0])
    explode_prob = np.array([0, 0.021, 0.042, 0.063, 0.146, 0.239, 0.313, 0.438, 0.563, 0.688, 0.792, 0.896, 1.0])
    max_pump = 13

    # Test for RL model
    '''
    model=RLBart(max_pump,explode_prob,accu_reward)
    Q_0 = 1.0
    alpha = 1.05
    gamma=0.10
    Lambda = 0.02
    tau=8.0
    pumps,explode = model.generate_data(Q_0,alpha,gamma,Lambda,tau)
    print(pumps)
    print('###########')
    print(explode)
    '''
    # Test for EW model
    '''
    model = EWBart(max_pump, explode_prob, accu_reward, const_subexplode_prob=False)
    psi = 0.05
    xi = 0.004
    rho = 1.0
    Lambda = 1.0
    tau = 5.0
    pumps, explode = model.generate_data(psi, xi, rho, Lambda, tau)
    print(pumps)
    print('###################')
    print(explode)
    '''
    '''
    # Test for STL model
    model = STLBart(max_pump,explode_prob,accu_reward)
    omega_0=0.5
    vwin = 0.1
    vloss=0.1
    tau = 8.0
    pumps,explode = model.generate_data(omega_0,vwin,vloss,tau)
    print(pumps)
    print('##############################')
    print(explode)
    '''

    '''
    # Test for New model
    model = NEWBart(max_pump, explode_prob, accu_reward)
    Q_0 = 1.0
    alpha = 0.1
    gamma = 0.5
    Lambda = 0.02
    tau = 8.0
    pumps, explode =model.generate_data(Q_0,alpha,gamma,Lambda,tau)
    print(pumps)
    print('##################################')
    print(explode)
    '''

    # Test for Four Parameter model
    model = FourparamBart(max_pump,explode_prob,accu_reward)
    phi = 0.9
    eta = 0.1
    gamma = 0.6
    tau = 8.0
    pumps, explode = model.generate_data(phi,eta,gamma,tau)
    print(pumps)
    print('########################')
    print(explode)
    # '''
    # # Validate old model
    # distributed = False
    # const_reward_all = [True, False]
    # const_subexplode_prob_all = [True, False]
    # penalty_all = [True, False]
    # model_name_all = ['EW', 'EWMV']
    # '''
    # # Test new model
    # distributed = False
    # const_reward = False
    # model_name = 'new'
    # trial_id = 'new_model'
    # '''
    # if distributed:
    #     idx = int(sys.argv[1])
    #     trial_id, const_reward, const_subexplode_prob, penalty, model_name = generate_option(const_reward_all,
    #                                                                                          const_subexplode_prob_all,
    #                                                                                          penalty_all,
    #                                                                                          model_name_all,
    #                                                                                          idx)
    # else:
    #     # Three options for model simulation:
    #     # constant reward: True or False
    #     const_reward = True
    #     # constant subjective explosion prpobability: True or False
    #     const_subexplode_prob = True
    #     # Penalty: Whether balloon explosion will be penalized, True or False
    #     penalty = False
    #     # model_name: FourParam, EW,EWMV
    #     model_name = 'FourParam'
    #     trial_id = 'FourParam'
    # '''
    # ####################################################################
    # if const_reward:
    #     accu_reward = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6])
    # else:
    #     accu_reward = np.array([0.0, 0.0, 0.05, 0.15, 0.25, 0.55, 0.95, 1.45, 2.05, 2.75, 3.45, 4.25, 5.15])
    #
    # config = {'trial_id': trial_id,
    #           'max_pump': 13,
    #           'explode_prob': np.array(
    #               [0, 0.021, 0.042, 0.063, 0.146, 0.239, 0.313, 0.438, 0.563, 0.688, 0.792, 0.896, 1.0]),
    #           'accu_reward': accu_reward,
    #           'model_name': model_name,
    #           # 'const_subexplode_prob': const_subexplode_prob,
    #           # 'penalty': penalty,
    #           }
    #
    # params_Fourparam = {'phi': np.linspace(0.95, 0.99, 5),
    #                     'eta': np.logspace(-7, -3.5, 8, base=2),
    #                     'gamma': np.linspace(0.3, 1.35, 8),
    #                     'tau': np.linspace(0.05, 0.75, 8)}
    #
    # params_EW = {'phi': np.linspace(0.01, 0.05, 5),
    #              'xi': np.logspace(-7, -3.5, 8, base=2),
    #              'rho': np.linspace(0.125, 1.0, 8),
    #              'Lambda': np.linspace(1.25, 10.0, 8),
    #              'tau': np.linspace(5, 20, 4)
    #              }
    #
    # params_EWMV = {'phi': np.linspace(0.01, 0.05, 5),
    #                'xi': np.logspace(-7, -3.5, 8, base=2),
    #                'rho': np.linspace(0.125, 1.0, 8),
    #                'Lambda': np.linspace(-0.004, 0.01, 8),
    #                'tau': np.linspace(5, 20, 4)
    #                }
    #
    # params_new = {'R_0': np.array([0.5]),
    #               'alpha': np.linspace(0.1, 0.9, 5),
    #               'gamma': np.linspace(0.5, 2.0, 4),
    #               'Lambda': np.linspace(0.02, 0.18, 5),
    #               'tau': np.linspace(5, 20, 4)}
    # if config['model_name'] == 'FourParam':
    #     params = params_Fourparam
    # elif config['model_name'] == 'EW':
    #     params = params_EW
    # elif config['model_name'] == 'EWMV':
    #     params = params_EWMV
    # elif config['model_name'] == 'new':
    #     params = params_new
    # else:
    #     raise ValueError('Invalid Model Name!')
    #
    # save_dir = 'simulation_data/'
    # # bart_simulation_main(config, save_dir,params)
    # # bart_simulation_new_main(config, save_dir,params)
    # # bart_generate_basic_info(config, save_dir)
    # # bart_plot_stop_main(config, save_dir)
    #
    # # bart_plot_pump_prob(config, save_dir)
    #
    # '''
    # model=EWBart(max_pump,explode_prob,reward)
    # phi=0.1
    # xi=1.0
    # rho=1.0
    # Lambda=0.5
    # tau=1.0
    # pumps,explode=model.generate_forward(xi,phi,rho,Lambda,tau)
    # print(np.mean(pumps))
    # plt.hist(pumps)
    # plt.show()
    # result=pd.DataFrame({'pumps':pumps,'explode':explode})
    # result.to_excel('simulation_result.xlsx',index=False)
    # '''
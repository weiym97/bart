import numpy as np
import pandas as pd

from bart_model import *


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


def PTBart_10_simulation_main(accu_reward, explode_prob, max_pump, params, data_dir, n_simu_subj, n_fit_per_run,
                              trial_per_subj=50):
    model = PTBart_10(max_pump=max_pump, accu_reward=accu_reward, explode_prob=explode_prob, num_trial=trial_per_subj)
    n_file = int(n_simu_subj / n_fit_per_run)
    result = []
    for i in range(n_simu_subj):
        pumps, explosion = model.generate_data(psi=params['psi'][i],
                                               xi=params['xi'][i],
                                               gamma=params['gamma'][i],
                                               tau=params['tau'][i],
                                               Lambda=params['lambda'][i])
        subjdata = pd.DataFrame({'subjID': params['subjID'][i],
                                 'group': 0,
                                 'trial': np.arange(trial_per_subj) + 1,
                                 'reward': accu_reward[pumps],
                                 'pumps': pumps,
                                 'explosion': explosion})
        result.append(subjdata)
    result = pd.concat(result)
    result.to_csv(data_dir + 'PTBART_10_simulation.txt', sep=' ', index=False, doublequote=False)
    for j in range(n_file):
        result_save = result.iloc[int(j * n_fit_per_run * trial_per_subj):int((j + 1) * n_fit_per_run * trial_per_subj),:]
        result_save.to_csv(data_dir + 'PTBart_10_simulation_'+str(j+1)+'.txt',sep=' ',index=False,doublequote=False)
    # return


if __name__ == '__main__':
    accu_reward = np.array([0.0, 0.0, 0.05, 0.15, 0.25, 0.55, 0.95, 1.45, 2.05, 2.75, 3.45, 4.25, 5.15, 6.0])
    explode_prob = np.array([0, 0.021, 0.042, 0.063, 0.146, 0.239, 0.313, 0.438, 0.563, 0.688, 0.792, 0.896, 1.0])
    max_pump = 13

    n_simu_subj = 3000
    n_fit_per_run = 100
    psi = np.random.uniform(0.02, 0.12, size=n_simu_subj)
    xi = 10 ** np.random.uniform(-3, -1, size=n_simu_subj)
    gamma = np.random.uniform(0.4, 1.0, size=n_simu_subj)
    Lambda = np.random.uniform(1.5, 3.5, size=n_simu_subj)
    tau = np.random.uniform(0.7, 2.5, size=n_simu_subj)
    params = pd.DataFrame({'subjID': np.arange(n_simu_subj) + 10001,
                           'psi': psi,
                           'xi': xi,
                           'gamma': gamma,
                           'lambda': Lambda,
                           'tau': tau})
    data_dir = 'data/simulation/'
    params.to_excel(data_dir + 'PTBart_10_simulation_statistics.xlsx', index=False)
    # config={'accu_reward':accu_reward,
    #        'explode_prob':explode_prob,
    #        'max_pump':max_pump,
    #        'params':params,
    #        }
    PTBart_10_simulation_main(accu_reward, explode_prob, max_pump, params, data_dir, n_simu_subj, n_fit_per_run)

    '''
    model = BASEBart_107(max_pump=max_pump,
                         accu_reward=accu_reward,
                         explode_prob=explode_prob)
    omega_0 = 0.45
    alpha = 0.05
    beta = 10.0
    theta = 0.8
    Lambda = 100.0
    tau = 4.0
    pumps,explode,omega_history,omega_loss_averse_history = model.generate_data(omega_0=omega_0,
                                                                                alpha=alpha,
                                                                                beta=beta,
                                                                                theta=theta,
                                                                                Lambda=Lambda,
                                                                                tau=tau,
                                                                                return_omega=True)
    result= pd.DataFrame({
        'pumps':pumps,
        'explode':explode,
        'omega_history':omega_history,
        'omega_loss_averse_history':omega_loss_averse_history,
    })
    print(result)
    print('######################')
    print(np.average(pumps))
    '''
    '''
    model = BASEBart_110(max_pump=max_pump,
                         accu_reward=accu_reward,
                         explode_prob=explode_prob)
    phi = 0.9
    eta = 0.01
    gamma = 0.4
    Lambda = 15.0
    theta = 0.8
    tau = 4.0
    pumps,explode,omega_history,omega_loss_averse_history = model.generate_data(phi=phi,
                                                                                eta=eta,
                                                                                gamma=gamma,
                                                                                Lambda=Lambda,
                                                                                theta=theta,
                                                                                tau=tau,
                                                                                return_omega = True
                                                                                )
    result= pd.DataFrame({
        'pumps':pumps,
        'explode':explode,
        'omega_history':omega_history,
        'omega_loss_averse_history':omega_loss_averse_history,
    })
    print(result)
    print('######################')
    print(np.average(pumps))
    '''
    '''
    model = PTBart_5(max_pump=max_pump,
                     accu_reward=accu_reward,
                    explode_prob=explode_prob)
    phi = 0.9
    eta = 0.005
    gamma = 0.5
    Lambda = 0.0
    tau = 4.0
    pumps,explode,omega_history = model.generate_data(phi=phi,
                                                      eta=eta,
                                                      gamma=gamma,
                                                      Lambda=Lambda,
                                                      tau=tau,
                                                      return_omega = True
                                                     )
    result= pd.DataFrame({
        'pumps':pumps,
        'explode':explode,
        'omega_history':omega_history,
        #'omega_loss_averse_history':omega_loss_averse_history,
    })
    print(pumps)
    print(explode)
    print(omega_history)
    '''
    '''
    model = PTBart_11(max_pump=max_pump,
                     accu_reward=accu_reward,
                     explode_prob=explode_prob)
    psi = 0.05
    xi = 0.005
    gamma = 0.5
    Lambda = 2.0
    tau = 4.0
    alpha = 1.0
    pumps,explode,omega_history,omega_original_history = model.generate_data(psi=psi,
                                                      xi=xi,
                                                      gamma=gamma,
                                                      Lambda=Lambda,
                                                      tau=tau,
                                                      alpha=alpha,
                                                      return_omega = True
                                                      )
    result= pd.DataFrame({
        'pumps':pumps,
        'explode':explode,
        'omega_history':omega_history,
        'omega_original_history':omega_original_history
        #'omega_loss_averse_history':omega_loss_averse_history,
    })
    print(pumps)
    print(explode)
    print(omega_history)
    print(omega_original_history)
    '''

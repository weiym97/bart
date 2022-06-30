import json

import numpy as np
import pandas as pd
from bart_model import *


def compute_likelihood_main(config,data,params):
    subjID = params['subjID'].unique()
    results=[]
    # Check whether subjID in data is the same as that in params
    if data['subjID'].unique().all() != subjID.all():
        raise ValueError('Different subjectID for data and params!')
    if config['model_name'] == 'RLBart':
        model = RLBart(max_pump = config['max_pump'],
                       explode_prob = config['explode_prob'],
                       accu_reward = config['accu_reward'],
                       model_type = config['model_type'])
        for subj in subjID:
            pumps = data[data['subjID'] == subj]['pumps'].to_numpy()
            explosion = data[data['subjID'] == subj]['explosion'].to_numpy()
            Q_0 = float(params[params['subjID'] == subj]['Q_0'])
            alpha = float(params[params['subjID'] == subj]['alpha'])
            gamma = float(params[params['subjID'] == subj]['gamma'])
            Lambda = float(params[params['subjID'] == subj]['lambda'])
            tau = float(params[params['subjID'] == subj]['tau'])
            neg_log_likelihood,Q_history = model.compute_likelihood(Q_0,alpha,gamma,Lambda,tau,pumps,explosion,return_Q=True)
            result={
                'subjID': subj,
                'neg_log_likelihood':neg_log_likelihood.tolist(),
                'Q_history': Q_history.tolist(),
                'pumps': pumps.tolist(),
                'explosion':explosion.tolist(),
            }
            results.append(pd.DataFrame(result))
        pd.concat(results).to_excel('analyze_result/RLBart_1.xlsx')
    elif config['model_name'] == 'STLBart':
        model = STLBart(max_pump = config['max_pump'],
                       explode_prob = config['explode_prob'],
                       accu_reward = config['accu_reward'],
                       )
        for subj in subjID:
            pumps = data[data['subjID'] == subj]['pumps'].to_numpy()
            explosion = data[data['subjID'] == subj]['explosion'].to_numpy()
            omega_0 = float(params[params['subjID'] == subj]['omega_0'])
            vwin = float(params[params['subjID'] == subj]['vwin'])
            vloss = float(params[params['subjID'] == subj]['vloss'])
            tau = float(params[params['subjID'] == subj]['tau'])
            neg_log_likelihood,omega_history = model.compute_likelihood(omega_0,vwin,vloss,tau,pumps,explosion,return_omega=True)
            result={
                'subjID':subj,
                'neg_log_likelihood':neg_log_likelihood.tolist(),
                'omega_history': omega_history.tolist(),
                'pumps': pumps.tolist(),
                'explosion':explosion.tolist(),
            }
            results.append(pd.DataFrame(result))
        pd.concat(results).to_excel('analyze_result/STLBart.xlsx')
    elif config['model_name'] == 'RLBart_0':
        model = RLBart_0(max_pump = config['max_pump'],
                       explode_prob = config['explode_prob'],
                       accu_reward = config['accu_reward'],
                       model_type = config['model_type'])
        for subj in subjID:
            pumps = data[data['subjID'] == subj]['pumps'].to_numpy()
            explosion = data[data['subjID'] == subj]['explosion'].to_numpy()
            Q_0 = float(params[params['subjID'] == subj]['Q_0'])
            alpha = float(params[params['subjID'] == subj]['alpha'])
            Lambda = float(params[params['subjID'] == subj]['lambda'])
            tau = float(params[params['subjID'] == subj]['tau'])
            neg_log_likelihood,Q_history = model.compute_likelihood(Q_0,alpha,Lambda,tau,pumps,explosion,return_Q=True)
            result={
                'subjID':subj,
                'neg_log_likelihood':neg_log_likelihood.tolist(),
                'Q_history': Q_history.tolist(),
                'pumps': pumps.tolist(),
                'explosion':explosion.tolist(),
            }
            results.append(pd.DataFrame(result))
        pd.concat(results).to_excel('analyze_result/RLBart_0.xlsx')
    elif config['model_name'] == "FourparamBart":
        model = FourparamBart(max_pump = config['max_pump'],
                        explode_prob = config['explode_prob'],
                        accu_reward = config['accu_reward'],
                        )
        for subj in subjID:
            pumps = data[data['subjID'] == subj]['pumps'].to_numpy()
            explosion = data[data['subjID'] == subj]['explosion'].to_numpy()
            phi = float(params[params['subjID'] == subj]['phi'])
            eta = float(params[params['subjID'] == subj]['eta'])
            gam = float(params[params['subjID'] == subj]['gam'])
            tau = float(params[params['subjID'] == subj]['tau'])
            neg_log_likelihood,omega_history = model.compute_likelihood(phi,eta,gam,tau,pumps,explosion,return_omega=True)
            result={
                'subjID':subj,
                'neg_log_likelihood':neg_log_likelihood.tolist(),
                'omega_history': omega_history.tolist(),
                'pumps': pumps.tolist(),
                'explosion':explosion.tolist(),
            }
            results.append(pd.DataFrame(result))
        pd.concat(results).to_excel('analyze_result/FourparamBart.xlsx')
    elif config['model_name'] == 'BASEBart_14':
        model = BASEBart_14(max_pump = config['max_pump'],
                            explode_prob = config['explode_prob'],
                            accu_reward = config['accu_reward'],
                              )
        for subj in subjID:
            pumps = data[data['subjID'] == subj]['pumps'].to_numpy()
            explosion = data[data['subjID'] == subj]['explosion'].to_numpy()
            omega_0 = float(params[params['subjID'] == subj]['omega_0'])
            alpha = float(params[params['subjID'] == subj]['alpha'])
            Lambda = float(params[params['subjID'] == subj]['lambda'])
            tau = float(params[params['subjID'] == subj]['tau'])
            neg_log_likelihood,omega_history = model.compute_likelihood(omega_0, alpha, Lambda,tau,pumps,explosion,return_omega=True)
            result={
                'subjID':subj,
                'neg_log_likelihood':neg_log_likelihood.tolist(),
                'omega_history': omega_history.tolist(),
                #'omega_history_times': omega_history.tolist()*config['max_pump'],
                'pumps': pumps.tolist(),
                'explosion':explosion.tolist(),
            }
            results.append(pd.DataFrame(result))
        pd.concat(results).to_excel('analyze_result/BASEBart_14.xlsx')
    elif config['model_name'] == 'BASEBart_30':
        model = BASEBart_30(max_pump = config['max_pump'],
                            explode_prob = config['explode_prob'],
                            accu_reward = config['accu_reward'],
                            )
        subjID =[1542]
        for subj in subjID:
            pumps = data[data['subjID'] == subj]['pumps'].to_numpy()
            explosion = data[data['subjID'] == subj]['explosion'].to_numpy()
            omega_0 = float(params[params['subjID'] == subj]['omega_0'])
            alpha = float(params[params['subjID'] == subj]['alpha'])
            beta = float(params[params['subjID'] == subj]['beta'])
            Lambda = float(params[params['subjID'] == subj]['lambda'])
            tau = float(params[params['subjID'] == subj]['tau'])
            omega_0 = 0.30
            alpha = 0.5
            beta = 0.5
            Lambda = -0.3
            tau = 0.73
            neg_log_likelihood,omega_history,omega_loss_aver_history = model.compute_likelihood(omega_0, alpha, beta, Lambda,tau,pumps,explosion,return_omega=True)
            result={
                'subjID':subj,
                'neg_log_likelihood':neg_log_likelihood.tolist(),
                'omega_history': omega_history.tolist(),
                'omega_loss_aver_history': omega_loss_aver_history.tolist(),
                'pumps': pumps.tolist(),
                'explosion':explosion.tolist(),
            }
            results.append(pd.DataFrame(result))
        pd.concat(results).to_excel('analyze_result/BASEBart_30_1542.xlsx')
    elif config['model_name'] == 'BASEBART_106':
        model = BASEBart_106(max_pump = config['max_pump'],
                            explode_prob = config['explode_prob'],
                            accu_reward = config['accu_reward'],
                            )
        #subjID =[1542]
        for subj in subjID:
            pumps = data[data['subjID'] == subj]['pumps'].to_numpy()
            explosion = data[data['subjID'] == subj]['explosion'].to_numpy()
            omega_0 = float(params[params['subjID'] == subj]['omega_0'])
            alpha = float(params[params['subjID'] == subj]['alpha'])
            beta = float(params[params['subjID'] == subj]['beta'])
            Lambda = float(params[params['subjID'] == subj]['lambda'])
            theta = float(params[params['subjID'] == subj]['theta'])
            tau = float(params[params['subjID'] == subj]['tau'])
            neg_log_likelihood,omega_history,omega_loss_aver_history = model.compute_likelihood(omega_0, alpha, beta, Lambda,tau,pumps,explosion,return_omega=True)
            result={
                'subjID':subj,
                'neg_log_likelihood':neg_log_likelihood.tolist(),
                'omega_history': omega_history.tolist(),
                'omega_loss_aver_history': omega_loss_aver_history.tolist(),
                'pumps': pumps.tolist(),
                'explosion':explosion.tolist(),
            }
            results.append(pd.DataFrame(result))
        pd.concat(results).to_excel('analyze_result/BASEBart_106_1542.xlsx')
    elif config['model_name'] == 'PTBart_10':
        model = PTBart_10(max_pump = config['max_pump'],
                          explode_prob = config['explode_prob'],
                          accu_reward = config['accu_reward'],
                         )
        for subj in subjID:
            pumps = data[data['subjID'] == subj]['pumps'].to_numpy()
            explosion = data[data['subjID'] == subj]['explosion'].to_numpy()
            psi = float(params[params['subjID'] == subj]['psi'])
            xi = float(params[params['subjID'] == subj]['xi'])
            gamma = float(params[params['subjID'] == subj]['gamma'])
            tau = float(params[params['subjID'] == subj]['tau'])
            Lambda = float(params[params['subjID'] == subj]['lambda'])
            neg_log_likelihood,omega_history = model.compute_likelihood(psi,xi,gamma,tau,Lambda,pumps,explosion,return_omega=True)
            result={
                'subjID':subj,
                'neg_log_likelihood':neg_log_likelihood.tolist(),
                'omega_history': omega_history.tolist(),
                'pumps': pumps.tolist(),
                'explosion':explosion.tolist(),
            }
            results.append(pd.DataFrame(result))
        pd.concat(results).to_excel('analyze_result/PTBart_10.xlsx')

    else:
        raise ValueError('Model under development!')

if __name__ == '__main__':
    # configuration
    accu_reward = np.array([0.0, 0.0, 0.05, 0.15, 0.25, 0.55, 0.95, 1.45, 2.05, 2.75, 3.45, 4.25, 5.15, 6.0])
    explode_prob = np.array([0, 0.021, 0.042, 0.063, 0.146, 0.239, 0.313, 0.438, 0.563, 0.688, 0.792, 0.896, 1.0])
    max_pump = 13
    '''
    config = {
        'max_pump': max_pump,
        'accu_reward': accu_reward,
        'explode_prob': explode_prob,
        'model_name': 'RLBart',
        'model_type': '1',
    }

    data = pd.read_csv('data/MDD_13.txt',sep='\t')
    params=pd.read_csv('fit_result/RLBart_1_MDD_13.csv')
    result = compute_likelihood_main(config,data,params)
    '''

    '''
    config = {
        'max_pump': max_pump,
        'accu_reward': accu_reward,
        'explode_prob': explode_prob,
        'model_name': 'STLBart',
    }

    data = pd.read_csv('data/MDD_13.txt',sep='\t')
    params=pd.read_csv('fit_result/STLBart_MDD_13.csv')
    result = compute_likelihood_main(config,data,params)
    '''
    '''
    config = {
        'max_pump': max_pump,
        'accu_reward': accu_reward,
        'explode_prob': explode_prob,
        'model_name': 'RLBart_0',
        'model_type': '1',
    }

    data = pd.read_csv('data/MDD_13.txt',sep='\t')
    params=pd.read_csv('fit_result/RLBart_0_MDD_13.csv')
    result = compute_likelihood_main(config,data,params)
    '''

    '''
    config = {
        'max_pump': max_pump,
        'accu_reward': accu_reward,
        'explode_prob': explode_prob,
        'model_name': 'FourparamBart',
        'model_type': '1',
    }

    data = pd.read_csv('data/MDD_13.txt',sep='\t')
    params=pd.read_csv('fit_result/FourparamBart_MDD_13.csv')
    result = compute_likelihood_main(config,data,params)
    '''
    '''
    config = {
        'max_pump': max_pump,
        'accu_reward': accu_reward,
        'explode_prob': explode_prob,
        'model_name': 'BASEBart_14',
        'model_type': '1',
    }

    data = pd.read_csv('data/MDD_13.txt',sep='\t')
    params=pd.read_csv('fit_result/BASEBart_14_MDD_13.csv')
    result = compute_likelihood_main(config,data,params)
    '''

    '''
    config = {
        'max_pump': max_pump,
        'accu_reward': accu_reward,
        'explode_prob': explode_prob,
        'model_name': 'BASEBart_30',
    }

    data = pd.read_csv('data/MDD_13.txt',sep='\t')
    params=pd.read_csv('fit_result/BASEBart_30_MDD_13.csv')
    result = compute_likelihood_main(config,data,params)
    '''

    config = {
        'max_pump': max_pump,
        'accu_reward': accu_reward,
        'explode_prob': explode_prob,
        'model_name': 'PTBart_10',
    }

    data = pd.read_csv('data/MDD_13_preprocessing.txt',sep=' ')
    params=pd.read_csv('fit_result/summary_PTBart_10_MDD_13.txt',sep=' ')
    result = compute_likelihood_main(config,data,params)
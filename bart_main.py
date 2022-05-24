def bart_simulation_new_main(config, save_dir, params):
    save_dir = save_dir + config['trial_id'] + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if config['model_name'] == 'FourParam':
        model = FourparamBart(max_pump=config['max_pump'],
                              explode_prob=config['explode_prob'],
                              accu_reward=config['accu_reward'],
                              const_subexplode_prob=config['const_subexplode_prob'],
                              penalty=config['penalty'],
                              )
        subject_ID = 1
        total_subject_ID = []
        total_phi = []
        total_eta = []
        total_gamma = []
        total_tau = []
        total_trial = []
        total_pumps = []
        total_explosion = []
        for phi in params['phi']:
            for eta in params['eta']:
                for gamma in params['gamma']:
                    for tau in params['tau']:
                        pumps, explode = model.generate_data(phi, eta, gamma, tau)
                        total_subject_ID.append(np.ones(len(pumps)) * subject_ID)
                        total_phi.append(np.ones(len(pumps)) * phi)
                        total_eta.append(np.ones(len(pumps)) * eta)
                        total_gamma.append(np.ones(len(pumps)) * gamma)
                        total_tau.append(np.ones(len(pumps)) * tau)
                        total_trial.append(np.arange(1, len(pumps) + 1))
                        total_pumps.append(pumps)
                        total_explosion.append(explode)

                        subject_ID += 1
        result = pd.DataFrame({'SubjID': np.concatenate(total_subject_ID),
                               'phi': np.concatenate(total_phi),
                               'eta': np.concatenate(total_eta),
                               'gamma': np.concatenate(total_gamma),
                               'tau': np.concatenate(total_tau),
                               'trial': np.concatenate(total_trial),
                               'pumps': np.concatenate(total_pumps),
                               'explosion': np.concatenate(total_explosion),
                               })

    elif config['model_name'] == 'EW':
        model = EWBart(max_pump=config['max_pump'],
                       explode_prob=config['explode_prob'],
                       accu_reward=config['accu_reward'],
                       const_subexplode_prob=config['const_subexplode_prob'],
                       penalty=config['penalty'],
                       )
        subject_ID = 1
        total_subject_ID = []
        total_phi = []
        total_xi = []
        total_rho = []
        total_Lambda = []
        total_tau = []
        total_trial = []
        total_pumps = []
        total_explosion = []
        for phi in params['phi']:
            for xi in params['xi']:
                for rho in params['rho']:
                    for Lambda in params['Lambda']:
                        for tau in params['tau']:
                            pumps, explode = model.generate_data(phi, xi, rho, Lambda, tau)
                            total_subject_ID.append(np.ones(len(pumps)) * subject_ID)
                            total_phi.append(np.ones(len(pumps)) * phi)
                            total_xi.append(np.ones(len(pumps)) * xi)
                            total_rho.append(np.ones(len(pumps)) * rho)
                            total_Lambda.append(np.ones(len(pumps)) * Lambda)
                            total_tau.append(np.ones(len(pumps)) * tau)
                            total_trial.append(np.arange(1, len(pumps) + 1))
                            total_pumps.append(pumps)
                            total_explosion.append(explode)

                        subject_ID += 1
        result = pd.DataFrame({'SubjID': np.concatenate(total_subject_ID),
                               'phi': np.concatenate(total_phi),
                               'xi': np.concatenate(total_xi),
                               'rho': np.concatenate(total_rho),
                               'Lambda': np.concatenate(total_Lambda),
                               'tau': np.concatenate(total_tau),
                               'trial': np.concatenate(total_trial),
                               'pumps': np.concatenate(total_pumps),
                               'explosion': np.concatenate(total_explosion),
                               })

    elif config['model_name'] == 'EWMV':
        model = EWMVBart(max_pump=config['max_pump'],
                         explode_prob=config['explode_prob'],
                         accu_reward=config['accu_reward'],
                         const_subexplode_prob=config['const_subexplode_prob'],
                         penalty=config['penalty'],
                         )
        subject_ID = 1
        total_subject_ID = []
        total_phi = []
        total_xi = []
        total_rho = []
        total_Lambda = []
        total_tau = []
        total_trial = []
        total_pumps = []
        total_explosion = []
        for phi in params['phi']:
            for xi in params['xi']:
                for rho in params['rho']:
                    for Lambda in params['Lambda']:
                        for tau in params['tau']:
                            pumps, explode = model.generate_data(phi, xi, rho, Lambda, tau)
                            total_subject_ID.append(np.ones(len(pumps)) * subject_ID)
                            total_phi.append(np.ones(len(pumps)) * phi)
                            total_xi.append(np.ones(len(pumps)) * xi)
                            total_rho.append(np.ones(len(pumps)) * rho)
                            total_Lambda.append(np.ones(len(pumps)) * Lambda)
                            total_tau.append(np.ones(len(pumps)) * tau)
                            total_trial.append(np.arange(1, len(pumps) + 1))
                            total_pumps.append(pumps)
                            total_explosion.append(explode)

                        subject_ID += 1
        result = pd.DataFrame({'SubjID': np.concatenate(total_subject_ID),
                               'phi': np.concatenate(total_phi),
                               'xi': np.concatenate(total_xi),
                               'rho': np.concatenate(total_rho),
                               'Lambda': np.concatenate(total_Lambda),
                               'tau': np.concatenate(total_tau),
                               'trial': np.concatenate(total_trial),
                               'pumps': np.concatenate(total_pumps),
                               'explosion': np.concatenate(total_explosion),
                               })

    elif config['model_name'] == 'new':
        model = NewBart(max_pump=config['max_pump'],
                        explode_prob=config['explode_prob'],
                        accu_reward=config['accu_reward'],
                        )
        subject_ID = 1
        total_subject_ID = []
        total_R_0 = []
        total_alpha = []
        total_gamma = []
        total_Lambda = []
        total_tau = []
        total_trial = []
        total_pumps = []
        total_explosion = []
        for R_0 in params['R_0']:
            for alpha in params['alpha']:
                for gamma in params['gamma']:
                    for Lambda in params['Lambda']:
                        for tau in params['tau']:
                            pumps, explode = model.generate_data(R_0, alpha, gamma, Lambda, tau)
                            total_subject_ID.append(np.ones(len(pumps)) * subject_ID)
                            total_R_0.append(np.ones(len(pumps)) * R_0)
                            total_alpha.append(np.ones(len(pumps)) * alpha)
                            total_gamma.append(np.ones(len(pumps)) * gamma)
                            total_Lambda.append(np.ones(len(pumps)) * Lambda)
                            total_tau.append(np.ones(len(pumps)) * tau)
                            total_trial.append(np.arange(1, len(pumps) + 1))
                            total_pumps.append(pumps)
                            total_explosion.append(explode)

                        subject_ID += 1
        result = pd.DataFrame({'SubjID': np.concatenate(total_subject_ID),
                               'R_0': np.concatenate(total_R_0),
                               'alpha': np.concatenate(total_alpha),
                               'gamma': np.concatenate(total_gamma),
                               'Lambda': np.concatenate(total_Lambda),
                               'tau': np.concatenate(total_tau),
                               'trial': np.concatenate(total_trial),
                               'pumps': np.concatenate(total_pumps),
                               'explosion': np.concatenate(total_explosion),
                               })
    else:
        raise ValueError('Invalid Model Name!')

    result.to_excel(save_dir + 'result.xlsx', index=False)

def bart_simulation_main(config, save_dir, params):
    save_dir = save_dir + config['trial_id'] + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if config['model_name'] == 'FourParam':
        model = FourparamBart(max_pump=config['max_pump'],
                              explode_prob=config['explode_prob'],
                              accu_reward=config['accu_reward'],
                              const_subexplode_prob=config['const_subexplode_prob'],
                              penalty=config['penalty'],
                              )
        subject_ID = 1
        total_subject_ID = []
        total_phi = []
        total_eta = []
        total_gamma = []
        total_tau = []
        total_trial = []
        total_pumps = []
        total_explosion = []
        for phi in params['phi']:
            for eta in params['eta']:
                for gamma in params['gamma']:
                    for tau in params['tau']:
                        pumps, explode = model.generate_data(phi, eta, gamma, tau)
                        total_subject_ID.append(np.ones(len(pumps)) * subject_ID)
                        total_phi.append(np.ones(len(pumps)) * phi)
                        total_eta.append(np.ones(len(pumps)) * eta)
                        total_gamma.append(np.ones(len(pumps)) * gamma)
                        total_tau.append(np.ones(len(pumps)) * tau)
                        total_trial.append(np.arange(1, len(pumps) + 1))
                        total_pumps.append(pumps)
                        total_explosion.append(explode)

                        subject_ID += 1
        result = pd.DataFrame({'SubjID': np.concatenate(total_subject_ID),
                               'phi': np.concatenate(total_phi),
                               'eta': np.concatenate(total_eta),
                               'gamma': np.concatenate(total_gamma),
                               'tau': np.concatenate(total_tau),
                               'trial': np.concatenate(total_trial),
                               'pumps': np.concatenate(total_pumps),
                               'explosion': np.concatenate(total_explosion),
                               })

    elif config['model_name'] == 'EW':
        model = EWBart(max_pump=config['max_pump'],
                       explode_prob=config['explode_prob'],
                       accu_reward=config['accu_reward'],
                       const_subexplode_prob=config['const_subexplode_prob'],
                       penalty=config['penalty'],
                       )
        subject_ID = 1
        total_subject_ID = []
        total_phi = []
        total_xi = []
        total_rho = []
        total_Lambda = []
        total_tau = []
        total_trial = []
        total_pumps = []
        total_explosion = []
        for phi in params['phi']:
            for xi in params['xi']:
                for rho in params['rho']:
                    for Lambda in params['Lambda']:
                        for tau in params['tau']:
                            pumps, explode = model.generate_data(phi, xi, rho, Lambda, tau)
                            total_subject_ID.append(np.ones(len(pumps)) * subject_ID)
                            total_phi.append(np.ones(len(pumps)) * phi)
                            total_xi.append(np.ones(len(pumps)) * xi)
                            total_rho.append(np.ones(len(pumps)) * rho)
                            total_Lambda.append(np.ones(len(pumps)) * Lambda)
                            total_tau.append(np.ones(len(pumps)) * tau)
                            total_trial.append(np.arange(1, len(pumps) + 1))
                            total_pumps.append(pumps)
                            total_explosion.append(explode)

                        subject_ID += 1
        result = pd.DataFrame({'SubjID': np.concatenate(total_subject_ID),
                               'phi': np.concatenate(total_phi),
                               'xi': np.concatenate(total_xi),
                               'rho': np.concatenate(total_rho),
                               'Lambda': np.concatenate(total_Lambda),
                               'tau': np.concatenate(total_tau),
                               'trial': np.concatenate(total_trial),
                               'pumps': np.concatenate(total_pumps),
                               'explosion': np.concatenate(total_explosion),
                               })

    elif config['model_name'] == 'EWMV':
        model = EWMVBart(max_pump=config['max_pump'],
                         explode_prob=config['explode_prob'],
                         accu_reward=config['accu_reward'],
                         const_subexplode_prob=config['const_subexplode_prob'],
                         penalty=config['penalty'],
                         )
        subject_ID = 1
        total_subject_ID = []
        total_phi = []
        total_xi = []
        total_rho = []
        total_Lambda = []
        total_tau = []
        total_trial = []
        total_pumps = []
        total_explosion = []
        for phi in params['phi']:
            for xi in params['xi']:
                for rho in params['rho']:
                    for Lambda in params['Lambda']:
                        for tau in params['tau']:
                            pumps, explode = model.generate_data(phi, xi, rho, Lambda, tau)
                            total_subject_ID.append(np.ones(len(pumps)) * subject_ID)
                            total_phi.append(np.ones(len(pumps)) * phi)
                            total_xi.append(np.ones(len(pumps)) * xi)
                            total_rho.append(np.ones(len(pumps)) * rho)
                            total_Lambda.append(np.ones(len(pumps)) * Lambda)
                            total_tau.append(np.ones(len(pumps)) * tau)
                            total_trial.append(np.arange(1, len(pumps) + 1))
                            total_pumps.append(pumps)
                            total_explosion.append(explode)

                        subject_ID += 1
        result = pd.DataFrame({'SubjID': np.concatenate(total_subject_ID),
                               'phi': np.concatenate(total_phi),
                               'xi': np.concatenate(total_xi),
                               'rho': np.concatenate(total_rho),
                               'Lambda': np.concatenate(total_Lambda),
                               'tau': np.concatenate(total_tau),
                               'trial': np.concatenate(total_trial),
                               'pumps': np.concatenate(total_pumps),
                               'explosion': np.concatenate(total_explosion),
                               })

    else:
        raise ValueError('Invalid Model Name!')

    result.to_excel(save_dir + 'result.xlsx', index=False)
import numpy as np
import matplotlib.pyplot as plt
from bart_model import RLBart


def RL_model_diagram(alpha,gamma,r_k,Q_k):
    for i in range(len(gamma)):
        figure1=plt.figure(1)
        for j in range(len(alpha)):
            Q_k_new=r_k-alpha[j]*(r_k-Q_k)+gamma[i]
            plt.plot(r_k,Q_k_new,label=r'$\alpha=$'+str(alpha[j])+','+r'$\gamma=$'+str(gamma[i]))
        plt.xlabel(r'$r_k$')
        plt.ylabel(r'$Q_{k+1}$')
        plt.legend()
        plt.title('Recursive formula '+r'$Q_k=$'+str(Q_k))
        plt.show()






if __name__ == '__main__':
    '''
    alpha = np.array([-0.25,0.0,0.25,0.5,0.75,1.0,1.25])
    gamma = np.array([-0.25,0.0,0.25])
    r_k = np.array([0.15,0.35,0.55,0.75,0.95,1.15,1.35])
    Q_k=0.75
    RL_model_diagram(alpha,gamma,r_k,Q_k)
    '''

    accu_reward = np.array([0.0, 0.0, 0.05, 0.15, 0.25, 0.55, 0.95, 1.45, 2.05, 2.75, 3.45, 4.25, 5.15, 6.0])
    explode_prob = np.array([0, 0.021, 0.042, 0.063, 0.146, 0.239, 0.313, 0.438, 0.563, 0.688, 0.792, 0.896, 1.0])
    max_pump = 13
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
    alphas=np.array([0.0,0.25,0.5,0.75,1.0,1.25])
    for alpha in alphas:
        pumps,explode,Q_history = model.generate_data(Q_0,alpha,gamma,Lambda,tau,return_Q=True)
        actual_reward=np.zeros_like(pumps)
        for j in range(len(pumps)):
            actual_reward[j] = accu_reward[int(pumps[j])]*(explode[j]*2-1)
        plt.plot(actual_reward,label='actual_reward')
        plt.plot(Q_history,label='Q_history')
        plt.legend()
        plt.title(r'$\alpha=$'+str(alpha))
        plt.show()
    '''
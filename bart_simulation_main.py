import numpy as np
import pandas as pd


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
    config = {

    }
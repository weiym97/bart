import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == '__main__':
    params = pd.read_csv('fit_result/FourparamBart_MDD_13.csv')
    subjID = params['subjID'].unique()
    for subj in subjID:
        with open('analyze_result/FourparamBart_'+str(subj)+'.json',"r") as f:
            result=json.load(f)
        print(result.keys())
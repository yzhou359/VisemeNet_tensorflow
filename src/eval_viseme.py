import numpy as np
import scipy as sp
from scipy import signal
import matplotlib.pyplot as plt
import copy
from src.utl.load_param import *

def eval_viseme(test_audio_name):

    def smooth(x, window_len=21, window='hanning'):

        if window_len < 3:
            return x

        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise(ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

        s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
        # print(len(s))
        if window == 'flat':  # moving average
            w = np.ones(window_len, 'd')
        else:
            w = eval('np.' + window + '(window_len)')

        y = np.convolve(w / w.sum(), s, mode='valid')
        return y

    src = pred_dir + test_audio_name + '/mayaparam'

    ac = np.loadtxt(src + '_pred_cls.txt')
    ar = np.loadtxt(src + '_pred_reg.txt')

    for i in range(2, ac.shape[1]):
        ac[2:-3, i] = sp.signal.medfilt(ac[2:-3, i], kernel_size=[9])
        ac[:, i] = smooth(ac[:, i], 9)[4:-4]
        ar[:, i] = sp.signal.medfilt(ar[:, i], kernel_size=[9])
        ar[:, i] = smooth(ar[:, i], 9)[4:-4]

    name_list = ['Ah', 'Aa', 'Eh', 'Ee', 'Ih', 'Oh', 'Uh', 'U', 'Eu', 'Schwa', 'R', 'S', 'ShChZh', 'Th',
                     'JY', 'LNTD', 'GK', 'MBP', 'FV', 'W']

    #pho_thd = np.array([0.35, 0.23, 0.18, 0.17, 10, 0.19, 0.18, 0.19, 10, 0.16,
    #                    0.18, 0.29, 0.29, 0.27, 10, 10, 10, 0.004, 0.29, 0.16]) # perfect

    pho_thd = np.array([0.12, 0.23, 0.18, 0.02, 10, 0.19, 0.18, 0.05, 10, 0.16,
                        0.18, 0.29, 0.29, 0.27, 10, 10, 10, 0.004, 0.29, 0.16])

    nb = np.zeros_like(ac)
    nb[:, 0] = smooth(ac[:, 0], 15)[7:-7]
    nb[:, 1] = smooth(ac[:, 1], 15)[7:-7]

    for i in range (2, ac.shape[1]):
        # times ac and ar
        tmp = ac[:, i] * ar[:, i]
        # print(pho_thd[i-2])
        l_idx = tmp > pho_thd[i-2]
        nb[l_idx, i] = ar[l_idx, i]

        nb[:, i] = smooth(nb[:, i], 15)[7:-7]

        r = 0
        while r < nb.shape[0]:
            if nb[r, i] > 0.1:
                active_begin = r
                for r2 in range(r, nb.shape[0]):
                    if nb[r2, i] < 0.1 or r2 == nb.shape[0] - 1:
                        active_end = r2
                        r = r2
                        break
                # print(active_begin, active_end)
                if (active_begin == active_end):
                    break
                max_reg = np.max(ar[active_begin:active_end, i])
                max_pred = np.max(nb[active_begin:active_end, i])
                rate = max_reg / max_pred
                nb[active_begin:active_end, i] = nb[active_begin:active_end, i] * rate
            r += 1
        nb[:, i] = smooth(nb[:, i], 15)[7:-7]

        r = 0
        while r < nb.shape[0]:
            if nb[r, i] > 0.1:
                active_begin = r
                for r2 in range(r, nb.shape[0]):
                    if nb[r2, i] < 0.1 or r2 == nb.shape[0] - 1:
                        active_end = r2
                        r = r2
                        break
                # print(active_begin, active_end)
                max_reg = np.max(ar[active_begin:active_end, i])
                if(i==19 or i==20 or i==21):
                    if(max_reg>0.7):
                        max_reg = 1
                max_pred = np.max(nb[active_begin:active_end, i])
                rate = max_reg / max_pred
                nb[active_begin:active_end, i] = nb[active_begin:active_end, i] * rate
            r += 1

    np.savetxt(src + '_viseme.txt',nb, '%.4f')

    print('Create Viseme parameter in ' + pred_dir + test_audio_name[:-4] + '/mayaparam_viseme.txt')
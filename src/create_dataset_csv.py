from src.utl.load_param import *
from src.utl.utl import try_mkdir
import numpy as np
import os, math
import scipy.io.wavfile as wav
from python_speech_features import logfbank, mfcc, ssc



def load_filelist_from_split_list(ver, dataset_type):

    """
    Load Youtube dataset file list from data_split dir
    :param dataset_type: choose all/train/cv/test for sub dataset
    :return: file_list
    """

    dir = data_split_dir
    file_list = {'landmark': [], 'phoneme': [], 'wav': []}

    for key in ['landmark', 'phoneme', 'wav']:
        txt = open(dir + dataset_type + '_' + key + '.txt')
        lines = txt.readlines()
        line_post = []
        for line in lines:
            if(is_up_sample and key == 'phoneme'):
                line_post.append(lpw_dir + line[:-5] + '_' + str(up_sample_rate) + '.txt')
            else:
                line_post.append(lpw_dir + line[:-1])
        file_list[key] = line_post
    file_list['n'] = len(file_list['wav'])

    return file_list


def create_dataset_csv(csv_dir, test_audio_name='test_audio.wav'):
    loaded_data = dict()
    loaded_data['wav'] = []
    loaded_data['phoneme'] = []
    loaded_data['landmark'] = []
    loaded_data['maya_pos'] = []
    loaded_data['maya_param'] = []
    loaded_data['face_close'] = []
    loaded_data['face_open'] = []
    loaded_data['pose'] = []
    loaded_data['file_len'] = {'train':0, 'test':0}
    loaded_data['clip_len'] = {'train':[], 'test':[]}
    loaded_data['file_dir'] = {'train':[], 'test':[]}
    dataset_type_order = ['test']

    csv_dir += test_audio_name[:-4] + '/'
    try_mkdir(csv_dir)
    try_mkdir(csv_dir + 'test/')
    errf = open(csv_dir + 'err.txt', 'w')

    for dataset_type_i in range(0,1):                            # all from train file list
        dataset_type = dataset_type_order[dataset_type_i]

        file_list = {'n':1, 'wav':[lpw_dir+test_audio_name]}

        for nClip in range(0, file_list['n']):

            print('\n==================== Processing file {:} ===================='.format(file_list["wav"][nClip]))
            if (not os.path.isfile(file_list["wav"][nClip])):
                print('# ' + str(nClip) + ' None existing file: ' + file_list["wav"][nClip])
                errf.write('# ' + str(nClip) + ' None existing file: ' + file_list["wav"][nClip] + '\n')
                continue

            # WAV
            (rate, sig) = wav.read(file_list["wav"][nClip])
            if (sig.ndim > 1):
                sig = sig[:, 0]  # pick mono-acoustic track
            else:
                print('Notice: ' + file_list["wav"][nClip] + ' is mono-track')

            # fps = (nLandmark + 1) / (sig.shape[0] / rate)
            fps = 25
            errf.write(file_list["wav"][nClip] + 'FPS: {:} \n'.format(fps))
            print('FPS: {:}'.format(fps))
            winstep = 1.0 / fps / mfcc_win_step_per_frame / up_sample_rate
            mfcc_feat = mfcc(sig, samplerate=rate, winlen=0.025, winstep=winstep, numcep=13)
            logfbank_feat = logfbank(sig, samplerate=rate, winlen=0.025, winstep=winstep, nfilt=26)
            ssc_feat = ssc(sig, samplerate=rate, winlen=0.025, winstep=winstep, nfilt=26)
            full_feat = np.concatenate([mfcc_feat, logfbank_feat, ssc_feat], axis=1)
            # full_feat = logfbank_feat

            nFrames_represented_by_wav = math.floor(full_feat.shape[0] / mfcc_win_step_per_frame / up_sample_rate)
            mfcc_lines = full_feat[0: nFrames_represented_by_wav * mfcc_win_step_per_frame * up_sample_rate, :].reshape(
                int(nFrames_represented_by_wav * up_sample_rate),
                int(full_feat.shape[1] * mfcc_win_step_per_frame))

            '''
            # ==================== cut the tail of lpw to make sure they are in same length ==================== #
            '''
            # print("Original length of lpw + maya_param/pos: " + str(nFrames_represented_by_wav))
            aligned_length_wav = mfcc_lines

            '''
            # ==================== process each lpw file ==================== #
            '''

            npWav = np.array(aligned_length_wav)
            print("Load #Clip {:d}/{:}, wav {:}".format(nClip, file_list['n'], npWav.shape))
            loaded_data['wav'].append(npWav)

            # length of each dataset_type
            loaded_data['file_len'][dataset_type] += npWav.shape[0]
            loaded_data['clip_len'][dataset_type].append(npWav.shape[0])
            loaded_data['file_dir'][dataset_type].append(file_list["wav"][nClip][28:-4]
                                                         + ' ' + str(loaded_data['file_len'][dataset_type] - npWav.shape[0])
                                                         + ' ' + str(npWav.shape[0]))
            # end for nClip loop
            # break

        # end for dataset_type loop
        # break

    '''
    # ==================== save file ==================== #
    '''
    key_order = ['wav']
    for key_i in range(0, 1):
        key = key_order[key_i]
       #  print(key)
        # ==================== wav normalize file ==================== #
        npKey = loaded_data[key][0]
        for i in range(1, len(loaded_data[key])):
            npKey = np.concatenate((npKey, loaded_data[key][i]), axis=0)

        # Use saved std & mean
        mean_std = np.loadtxt(lpw_dir + '/wav_mean_std.csv')
        npKey_mean = mean_std[0:65]
        npKey_std = mean_std[65:130]

        def normal_data(loaded_data, mean, std):
            normed = (loaded_data - mean) / std
            return normed

        npKey = normal_data(npKey, npKey_mean, npKey_std)
        np.savetxt(csv_dir + key + '_mean_std.csv', np.append(npKey_mean, npKey_std), fmt='%.5f', delimiter=' ')
        np.savetxt(csv_dir + key + '_raw.csv', npKey, fmt='%.5f', delimiter=' ')
        del npKey

        def reshape_based_on_win_size(loaded_data, i, win_size, start_idx):
            npWav = (loaded_data[i] - npKey_mean) / npKey_std
            listWav = list(range(start_idx, start_idx + npWav.shape[0]))
            half_win_size = int(win_size / 2)
            pad_head = [start_idx for _ in range(half_win_size)]
            pad_tail = [listWav[-1] for _ in range(half_win_size)]
            pad_npWav = np.array(pad_head + listWav + pad_tail)
            npKey = np.zeros(shape=(npWav.shape[0], win_size))
            for np_i in range(0, npWav.shape[0]):
                npKey[np_i] = pad_npWav[np_i:np_i + win_size].reshape(1, win_size)
            return npKey

        npKey = reshape_based_on_win_size(loaded_data['wav'], 0, win_size, 0)

        for i in range(1, len(loaded_data[key])):
            npKeytmp = reshape_based_on_win_size(loaded_data['wav'], i, win_size, npKey.shape[0])
            npKey = np.concatenate((npKey, npKeytmp), axis=0)

        idx = 0
        for dataset_type_i in range(0, 1):
            dataset_type = dataset_type_order[dataset_type_i]
            dataset_type_data_len = loaded_data['file_len'][dataset_type]
            cur_npKey = npKey[idx:idx+dataset_type_data_len]
            print('Save {:} - {:} file as shape of {:}'.format(dataset_type, key, cur_npKey.shape))
            np.savetxt(csv_dir + dataset_type + '/' + key + '.csv', cur_npKey, fmt='%d', delimiter=' ')
            idx += dataset_type_data_len

    for dataset_type in {'test'}:
        npLen = np.array(loaded_data['clip_len'][dataset_type])
        np.savetxt(csv_dir + dataset_type + '/clip_len.csv', npLen, fmt='%d', delimiter=' ')
        # print("Saved clip length file to " + dataset_type + '/clip_len.csv')
        npLen = np.array(loaded_data['file_dir'][dataset_type])
        np.savetxt(csv_dir + dataset_type + '/file_dir.csv', npLen, fmt='%s', delimiter=' ')
        # print("Saved file dir file to " + dataset_type + '/file_dir.csv')


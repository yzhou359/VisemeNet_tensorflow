import numpy as np
import tensorflow as tf
import glob

from src.utl.load_param import *
from src.model import model
from src.utl.load_param import *
from src.utl.utl import *
import math
import time


def test(model_name, test_audio_name):

    csv_test_audio = csv_dir + test_audio_name + '/'

    init, net1_optim, net2_optim, all_optim, x, x_face_id, y_landmark, y_phoneme, y_lipS, y_maya_param, dropout, cost, \
    tensorboard_op, pred, clear_op, inc_op, avg, batch_size_placeholder, phase = model()

    # start tf graph
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    max_to_keep = 20
    saver = tf.train.Saver(max_to_keep=max_to_keep)


    try_mkdir(pred_dir)

    # Test sess, load ckpt
    OLD_CHECKPOINT_FILE = model_dir + model_name + '/' + model_name +'.ckpt'

    saver.restore(sess, OLD_CHECKPOINT_FILE)
    print("Model loaded: " + model_dir + model_name)

    total_epoch_num = 1
    print(csv_test_audio)

    data_dir = {'train': {}, 'test': {}}
    data_dir['test']['wav'] = open(csv_test_audio + "test/wav.csv", 'r')
    data_dir['test']['clip_len'] = open(csv_test_audio + "test/clip_len.csv", 'r')
    cv_file_len = simple_read_clip_len(data_dir['test']['clip_len'])
    print('Loading wav_raw.txt file in {:}'.format(csv_test_audio))

    train_wav_raw = np.loadtxt(csv_test_audio + 'wav_raw.csv')
    test_wav_raw = train_wav_raw


    for epoch in range(0, total_epoch_num):
        # clear data file header

        # ============================== TRAIN SET CHUNK ITERATION ============================== #

        sess.run(clear_op)
        for key in ['train', 'test']:
            for lpw_key in data_dir[key].keys():
                data_dir[key][lpw_key].seek(0)

        print("===================== TEST/CV CHUNK - {:} ======================".format(csv_test_audio))
        eof = False
        chunk_num = 0
        chunk_size_sum = 0

        batch_size = test_wav_raw.shape[0]
        chunk_size = batch_size * batch_per_chunk_size

        while (not eof):
            cv_data, eof = read_chunk_data(data_dir, 'test', chunk_size)
            chunk_num += 1
            chunk_size_sum += len(cv_data['wav'])

            print('Load Chunk {:d}, size {:d}, total_size {:d} ({:2.2f})'
                  .format(chunk_num, len(cv_data['wav']), chunk_size_sum, chunk_size_sum / cv_file_len))

            full_idx_array = np.arange(len(cv_data['wav']))
            # np.random.shuffle(full_idx_array)
            for next_idx in range(0, int(np.floor(len(cv_data['wav']) / batch_size))):
                batch_idx_array = full_idx_array[next_idx * batch_size: (next_idx + 1) * batch_size]
                batch_x, batch_x_face_id, batch_x_pose, batch_y_landmark, batch_y_phoneme, batch_y_lipS, batch_y_maya_param = \
                    read_next_batch_easy_from_raw(test_wav_raw, cv_data, 'face_close', batch_idx_array, batch_size, n_steps, n_input, n_landmark,
                                         n_phoneme, n_face_id)
                npClose = np.loadtxt(lpw_dir + 'saved_param/maya_close_face.txt')
                batch_x_face_id = np.tile(npClose, (batch_x_face_id.shape[0], 1))


                test_pred, loss, _ = sess.run([pred, cost, inc_op],
                                            feed_dict={x: batch_x,
                                                       x_face_id: batch_x_face_id,
                                                       y_landmark: batch_y_landmark,
                                                       y_phoneme: batch_y_phoneme,
                                                       y_lipS: batch_y_lipS,
                                                       dropout: 0,
                                                       batch_size_placeholder: batch_x.shape[0],
                                                       phase: 0,
                                                       y_maya_param: batch_y_maya_param})


                def save_output(filename, npTxt, fmt):
                    f = open(filename, 'wb')
                    np.savetxt(f, npTxt, fmt=fmt)
                    f.close()

                try_mkdir(pred_dir + test_audio_name)

                def sigmoid(x):
                    return 1/(1+np.exp(-x))
                save_output(pred_dir + test_audio_name + "/mayaparam_pred_cls.txt",
                            np.concatenate([test_pred['jali'], sigmoid(test_pred['v_cls'])], axis=1), '%.4f')
                save_output(pred_dir + test_audio_name + "/mayaparam_pred_reg.txt",
                            np.concatenate([test_pred['jali'], test_pred['v_reg']], axis=1), '%.4f')

import os, sys
import numpy as np

def try_mkdir(dir, warning=True):
    try:
        os.makedirs(dir)
    except FileExistsError:
        if(warning):
            print("Warning: dir " + dir + " already exist! Continue program...")
    except:
        print("Cannot make dir: " + dir)
        print(sys.exc_info()[0])
        exit(0)


def simple_read_clip_len(filehead):
    lines = 0
    for line in filehead:
        # print(line)
        lines += int(line)
    # print(lines)
    return lines


def read_chunk_as_float_from_file(file_head, chunk_size):
    chunk_float = []
    eof = False
    for i in range(0, chunk_size):
        line = file_head.readline()
        number_str = line.split()
        number_float = [float(x) for x in number_str]
        if number_float == []:
            eof = True
            break
        chunk_float.append(number_float)
    return chunk_float, eof


def read_chunk_data(data_dir, data_type, chunk_size):
    data = dict()
    eof = False
    for lpw_key in data_dir[data_type].keys():
        data[lpw_key], e = read_chunk_as_float_from_file(data_dir[data_type][lpw_key], chunk_size)
        if(not eof and e and lpw_key=='wav'):
            eof = True

    return data, eof



def read_next_batch_easy_from_raw(wav_raw, data, face_type, batch_idx_array, batch_size, n_steps, n_input, n_landmark,
                                  n_phoneme, n_face_id):
    batch_x = np.zeros((batch_size, n_steps, n_input))
    batch_x_face_id = np.zeros((batch_size, n_face_id))
    batch_x_pose = np.zeros((batch_size, 3))
    batch_y_landmark = np.zeros((batch_size, n_landmark))
    batch_y_phoneme = np.zeros((batch_size, n_phoneme))
    batch_y_lipS = np.zeros((batch_size, 1))
    batch_y_maya_param = np.zeros((batch_size, 22))

    for i in range(0, batch_size):
        idx = batch_idx_array[i]
        wav_idx = [int(i) for i in data['wav'][idx]]
        batch_x[i] = wav_raw[wav_idx].reshape((1, n_steps, n_input))

    return batch_x, batch_x_face_id, batch_x_pose, batch_y_landmark, batch_y_phoneme, batch_y_lipS, batch_y_maya_param




# Dataset Setting

fps = 25                   # video
isPCA = False # False                    # landmark
PCA_fracs = 20
landmark_mouth_only = True
mfcc_win_step_per_frame = 1     # wav
mfcc_dim = 26 + 26 + 13
up_sample_rate = 4
is_up_sample = True
hasPhoneme = True
hasLandmark = True
hasFaceID = True
hasMaya = True

kernel_type = 'lstm'

# kernel param setting
win_size = 24  # 8
n_layers = 3
n_steps = 8
n_input = int(mfcc_dim * mfcc_win_step_per_frame * win_size / n_steps)
n_hidden = 256

n_phoneme = 21
n_out_fc1 = 256
end_or_mid = 'e'
n_filter = 256

n_hidden_net2_jali = 128
n_hidden_net2_cls = 256
n_hidden_net2_reg = 256
n_cls_fc1 = 200
n_reg_fc1 = 200
n_jali_fc1 = 200

n_landmark = 76
n_face_id = 76
n_maya_param = 22

reg_lambda = 0.001

dropout_value = 0.5
learning_rate = 0.000001
batch_size = 128
batch_per_chunk_size = 1     #
chunk_size = batch_size * batch_per_chunk_size
total_epoch_num = 10000
save_epoch_num = 10
save_ckpt = True
check_cv_epoch_num = 1
frame_delay = 0
z_dim = 100

p_alpha = 0.3

win_size_2 = 64

# DIR setting
root_dir = "data/"
lpw_dir = root_dir + 'test_audio/'
csv_dir = root_dir + 'csv/'
model_dir = root_dir + 'ckpt/'
logs_dir = root_dir + 'logs/'
pred_dir = root_dir + 'output_viseme/'
pic_dir = root_dir + 'pic/'
mp4_dir = root_dir + 'mp4/'

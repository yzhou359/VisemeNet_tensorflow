import tensorflow as tf

from src.utl.load_param import *
import numpy as np
import numpy.random


def model():

    #  tf graph input
    with tf.name_scope('input'):
        batch_size_placeholder = tf.placeholder("float32")
        # net 1
        x = tf.placeholder("float32", [None, n_steps, n_input])
        x_face_id = tf.placeholder("float32", [None, n_face_id])
        y_landmark = tf.placeholder("float32", [None, n_landmark])
        y_phoneme = tf.placeholder("float32", [None, n_phoneme])
        y_lipS = tf.placeholder("float32", [None, 1])
        phase = tf.placeholder(tf.bool, name='phase')
        # net 2
        y_maya_param = tf.placeholder("float32", [None, n_maya_param])


    # fully connected layer weights and bias
    with tf.name_scope('net1_fc'):
        n_out_landmark_fc2 = n_landmark
        n_out_phoneme_fc2 = n_phoneme

        w1_land = tf.Variable(tf.concat(
            [tf.truncated_normal([n_hidden, n_out_fc1], stddev=2 / (n_hidden + n_out_fc1), dtype=tf.float32),
             tf.truncated_normal([n_face_id, n_out_fc1], stddev=2 / (n_face_id + n_out_fc1), dtype=tf.float32)],
            axis=0), name='net1_w1_land')
        w1_pho = tf.Variable(
            tf.truncated_normal([n_hidden + n_face_id, n_out_fc1], stddev=2 / (n_hidden + n_face_id + n_out_fc1),
                                dtype=tf.float32), name='net1_w1_pho')

        w2_land = tf.Variable(
            tf.truncated_normal([n_out_fc1, n_out_landmark_fc2], stddev=2 / (n_out_fc1 + n_out_landmark_fc2),
                                dtype=tf.float32), name='net1_w2_land')
        w2_pho = tf.Variable(
            tf.truncated_normal([n_out_fc1, n_out_phoneme_fc2], stddev=2 / (n_out_fc1 + n_out_phoneme_fc2),
                                dtype=tf.float32), name='net1_w2_pho')

        b1_land = tf.Variable(tf.ones([n_out_fc1], dtype=tf.float32) * 0.1, name='net1_b1_land')
        b2_land = tf.Variable(tf.zeros([n_out_landmark_fc2], dtype=tf.float32), name='net1_b2_land')
        b1_pho = tf.Variable(tf.ones([n_out_fc1], dtype=tf.float32) * 0.1, name='net1_b1_pho')
        b2_pho = tf.Variable(tf.zeros([n_out_phoneme_fc2], dtype=tf.float32), name='net1_b2_pho')


    # LSTM model
    with tf.name_scope('net1_shared_rnn'):
        dropout = tf.placeholder("float32")

        if (kernel_type == 'rnn'):
            cell_func = tf.contrib.rnn.BasicRNNCell
        elif (kernel_type == 'lstm'):
            # cell_func = tf.contrib.rnn.BasicLSTMCell
            cell_func = tf.contrib.rnn.LSTMCell
        elif (kernel_type == 'gru'):
            cell_func = tf.contrib.rnn.GRUCell

        def one_layer_lstm_kernel(x, dropout, n_hidden):
            lstm_cell = cell_func(n_hidden, initializer=tf.glorot_normal_initializer())
            cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=1.0 - dropout)
            return tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

        def n_layer_rnn_kernel(x, dropout, n_layers, n_hidden):
            cells = []
            for _ in range(n_layers):
                lstm_cell = cell_func(n_hidden)
                lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=1.0 - dropout)
                cells.append(lstm_cell)
            cell = tf.contrib.rnn.MultiRNNCell(cells)
            return tf.nn.dynamic_rnn(cell, x, dtype=tf.float32, scope='net1_rnn')

        net1_rnn_output, states = n_layer_rnn_kernel(x=x, dropout=dropout, n_layers=n_layers,
                                                n_hidden=n_hidden)  # x in [n_batch x n_step x n_feature]
        # outputs = net1_rnn_output

    with tf.name_scope('net1_output'):
        outputs = net1_rnn_output[:, -1, :]
        outputs = tf.concat([outputs, x_face_id], axis=1)

        pred = dict()
        l1 = tf.matmul(outputs, w1_land) + b1_land
        l2 = tf.contrib.layers.batch_norm(l1, center=True, scale=True, is_training=phase, scope='net1_land_bn')
        l3 = tf.nn.relu(l2, name='net1_land_relu')
        l4 = tf.matmul(l3, w2_land) + b2_land
        pred['net1_land'] = l4 + x_face_id

        p1 = tf.matmul(outputs, w1_pho) + b1_pho
        p2 = tf.contrib.layers.batch_norm(p1, center=True, scale=True, is_training=phase, scope='net1_pho_bn')
        p3 = tf.nn.relu(p2, name='net1_pho_relu')
        p4 = tf.matmul(p3, w2_pho) + b2_pho
        pred['net1_pho'] = p4

    # error
    with tf.name_scope('net1_pho_err'):
        mistakes = tf.not_equal(tf.argmax(y_phoneme, 1), tf.argmax(pred['net1_pho'], 1))
        net1_pho_err = tf.reduce_mean(tf.cast(mistakes, tf.float32))

    # =========================================  NET 2  =============================================================
    with tf.name_scope('net2_rnn'):

        net2_input = tf.concat([pred['net1_land'], p3], axis=1)
        n_net2_input = n_landmark + n_out_fc1

        net2_input_concat = tf.concat([net2_input, x[:, int(n_steps/2), :]], axis=1)
        n_net2_input = n_net2_input + n_input


        net2_input_concat = tf.concat([tf.zeros(shape=(win_size_2/2, n_net2_input), dtype=tf.float32),
                                       net2_input_concat], axis=0)

        net2_input = tf.map_fn(
            lambda i: tf.reshape(net2_input_concat[i:i + win_size_2], (n_steps, int(n_net2_input * win_size_2 / n_steps))),
            tf.range(tf.shape(net2_input_concat)[0] - int(win_size_2)),
            dtype=tf.float32)
        net2_input = tf.cast(net2_input, tf.float32)
        y_maya_param_in = y_maya_param[0 : tf.shape(net2_input)[0]]

        # Net2 LSTM
        cells_jali = []
        cells_cls = []
        cells_reg = []
        for _ in range(3):
            lstm_cell_jali = tf.contrib.rnn.LSTMCell(n_hidden_net2_jali)
            lstm_cell_jali = tf.contrib.rnn.DropoutWrapper(lstm_cell_jali, output_keep_prob=0.5)
            cells_jali.append(lstm_cell_jali)
        for _ in range(1):
            lstm_cell_cls = tf.contrib.rnn.LSTMCell(n_hidden_net2_cls)
            lstm_cell_cls = tf.contrib.rnn.DropoutWrapper(lstm_cell_cls, output_keep_prob=0.5)
            cells_cls.append(lstm_cell_cls)
        for _ in range(3):
            lstm_cell_reg = tf.contrib.rnn.LSTMCell(n_hidden_net2_reg)
            lstm_cell_reg = tf.contrib.rnn.DropoutWrapper(lstm_cell_reg, output_keep_prob=0.5)
            cells_reg.append(lstm_cell_reg)
        cell_jali = tf.contrib.rnn.MultiRNNCell(cells_jali)
        cell_cls = tf.contrib.rnn.MultiRNNCell(cells_cls)
        cell_reg = tf.contrib.rnn.MultiRNNCell(cells_reg)
        output_jali, _ = tf.nn.dynamic_rnn(cell_jali, net2_input, dtype=tf.float32, scope='net2_rnn_jali')
        output_cls, _ = tf.nn.dynamic_rnn(cell_cls, net2_input, dtype=tf.float32, scope='net2_rnn_cls')
        output_reg, _ = tf.nn.dynamic_rnn(cell_reg, net2_input, dtype=tf.float32, scope='net2_rnn_reg')
        output_jali = output_jali[:, -1, :]
        output_cls = output_cls[:, -1, :]
        output_reg = output_reg[:, -1, :]

    with tf.name_scope('net2_fc'):
        w1_cls = tf.Variable(tf.truncated_normal([n_hidden_net2_cls, n_cls_fc1], stddev=2 / (n_hidden_net2_cls + n_cls_fc1)),
                             dtype=tf.float32, name='net2_w1_cls')
        b1_cls = tf.Variable(tf.constant(0.1, shape=[n_cls_fc1]), dtype=tf.float32, name='net2_b1_cls')
        w2_cls = tf.Variable(
            tf.truncated_normal([n_cls_fc1, n_maya_param-2], stddev=2 / (n_cls_fc1 + n_maya_param-2)),
            dtype=tf.float32, name='net2_w2_cls')
        b2_cls = tf.Variable(tf.constant(0.1, shape=[n_maya_param-2]), dtype=tf.float32, name='net2_b2_cls')

        w1_reg = tf.Variable(tf.truncated_normal([n_hidden_net2_reg, n_reg_fc1], stddev=2 / (n_hidden_net2_reg + n_reg_fc1)),
                             dtype=tf.float32, name='net2_w1_reg')
        b1_reg = tf.Variable(tf.constant(0.1, shape=[n_reg_fc1]), dtype=tf.float32, name='net2_b1_reg')
        w2_reg = tf.Variable(
            tf.truncated_normal([n_reg_fc1, 100], stddev=2 / (n_reg_fc1 + 100)),
            dtype=tf.float32, name='net2_w2_reg')
        b2_reg = tf.Variable(tf.constant(0.1, shape=[100]), dtype=tf.float32, name='net2_b2_reg')
        w3_reg = tf.Variable(
            tf.truncated_normal([100, n_maya_param-2], stddev=2 / (100 + n_maya_param-2)),
            dtype=tf.float32, name='net2_w3_reg')
        b3_reg = tf.Variable(tf.constant(0.1, shape=[n_maya_param-2]), dtype=tf.float32, name='net2_b3_reg')

        w1_jali = tf.Variable(tf.truncated_normal([n_hidden_net2_jali, n_jali_fc1], stddev=2 / (n_hidden_net2_jali + n_jali_fc1)),
                              dtype=tf.float32, name='net2_w1_jali')
        b1_jali = tf.Variable(tf.constant(0.1, shape=[n_jali_fc1]), dtype=tf.float32, name='net2_b1_jali')
        w2_jali = tf.Variable(tf.truncated_normal([n_jali_fc1, 2], stddev=2 / (n_jali_fc1 + 2)),
                              dtype=tf.float32, name='net2_w2_jali')
        b2_jali = tf.Variable(tf.constant(0.1, shape=[2]), dtype=tf.float32, name='net2_b2_jali')

    with tf.name_scope('net2_output'):
        v1_cls = tf.matmul(output_cls, w1_cls) + b1_cls
        v2_cls = tf.contrib.layers.batch_norm(v1_cls, center=True, scale=True, is_training=phase, scope='net2_v_cls_bn')
        v3_cls = tf.nn.relu(v2_cls, name='net2_v_cls_relu')
        pred['v_cls'] = tf.matmul(v3_cls, w2_cls) + b2_cls

        v1_reg = tf.matmul(output_reg, w1_reg) + b1_reg
        v2_reg = tf.contrib.layers.batch_norm(v1_reg, center=True, scale=True, is_training=phase, scope='net2_v_reg_bn')
        v3_reg = tf.nn.relu(v2_reg, name='net2_v_reg_relu')
        v4_reg = tf.matmul(v3_reg, w2_reg) + b2_reg
        v5_reg = tf.contrib.layers.batch_norm(v4_reg, center=True, scale=True, is_training=phase, scope='net2_v_reg_bn2')
        v6_reg = tf.nn.relu(v5_reg, name='net2_v_reg_relu2')
        pred['v_reg'] = tf.matmul(v6_reg, w3_reg) + b3_reg

        j1 = tf.matmul(output_jali, w1_jali) + b1_jali
        j2 = tf.contrib.layers.batch_norm(j1, center=True, scale=True, is_training=phase, scope='net2_jali_bn')
        j3 = tf.nn.relu(j2, name='net2_jali_relu')
        pred['jali'] = tf.matmul(j3, w2_jali) + b2_jali


    # loss
    with tf.name_scope('net1_loss'):
        cost = dict()
        # LipS weight
        pred_diff = pred['net1_land'] - y_landmark
        tile_lipS = tf.tile(y_lipS, [1, n_landmark])
        weighted_pred_diff = tf.multiply(pred_diff, tile_lipS)

        odd_land_pred = tf.multiply(pred['net1_land'], tile_lipS)[1::2,:]
        even_land_pred = tf.multiply(pred['net1_land'], tile_lipS)[0::2,:]
        m_land_pred = odd_land_pred - even_land_pred

        cost['net1_motion'] = tf.reduce_mean(tf.abs(m_land_pred)) * 1000
        cost['net1_land'] = tf.reduce_mean(tf.abs(weighted_pred_diff)) * 1000
        cost['net1_pho'] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred['net1_pho'], labels=y_phoneme))
        cost['net1_pho_err'] = net1_pho_err

        t_vars = tf.trainable_variables()
        reg_losses_1 = [tf.reduce_sum(tf.nn.l2_loss(var)) for var in t_vars if ('net1_' in var.name)]
        cost['net1_regularization'] = sum(reg_losses_1) / len(reg_losses_1)

        cost['net1'] = cost['net1_land'] * 0.25 + cost['net1_pho'] * p_alpha + 0.01 * cost['net1_regularization'] + 0.01 * cost['net1_motion']

    with tf.name_scope('net2_loss'):
        # cost = dict()

        cond = tf.less(y_maya_param_in[:, 2:n_maya_param], 0.01 * tf.ones(tf.shape(y_maya_param_in[:, 2:n_maya_param])))
        mask = tf.where(cond, tf.zeros(tf.shape(y_maya_param_in[:, 2:n_maya_param])),
                        tf.ones(tf.shape(y_maya_param_in[:, 2:n_maya_param])))

        cost['net2_v_cls'] = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=mask, logits=pred['v_cls'])) / batch_size_placeholder
        cost['net2_v_reg'] = tf.reduce_sum(
            tf.abs(pred['v_reg'] * mask - y_maya_param_in[:, 2:n_maya_param] * mask)) / batch_size_placeholder

        cost['net2_jali'] = tf.reduce_sum(tf.abs(pred['jali'] - y_maya_param_in[:, 0:2])) / batch_size_placeholder

        odd_frame_y = y_maya_param_in[1::2, :]
        even_frame_y = y_maya_param_in[0::2, :]
        m_y = odd_frame_y - even_frame_y

        cls_sig = tf.sigmoid(pred['v_cls'], name='net2_v_cls_sigmoid')
        # pvv = cls_sig*pred['v_reg']
        pvv = cls_sig
        odd_frame_pred_phone = pvv[1::2, :]
        even_frame_pred_phone = pvv[0::2, :]
        m_pred_phone = odd_frame_pred_phone - even_frame_pred_phone

        odd_frame_pred_jali = pred['jali'][1::2, :]
        even_frame_pred_jali = pred['jali'][0::2, :]
        m_pred_jali = odd_frame_pred_jali - even_frame_pred_jali

        cost['net2_1st_deriv'] = tf.reduce_mean(tf.abs(m_pred_phone - m_y[:, 2:])) + tf.reduce_mean(tf.abs(m_pred_jali - m_y[:, 0:2]))
        cost['net2_motion'] = tf.reduce_mean(tf.abs(pred['v_reg'][1::2,:]-pred['v_reg'][0::2,:])) + \
                              tf.reduce_mean(tf.abs(pred['v_cls'][1::2, :] - pred['v_cls'][0::2, :]))

        pred['viseme'] = tf.cast((cls_sig > 0.5), dtype=tf.float32) * pred['v_reg']

        t_vars = tf.trainable_variables()
        reg_losses = [tf.reduce_sum(tf.abs(var)) for var in t_vars if ('net2_' in var.name and '_b' not in var.name)]

        cost['net2'] = 0.35 * cost['net2_v_cls'] * p_alpha + 0.2 * cost['net2_v_reg'] + 0.2 * cost['net2_jali'] \
                       + cost['net2_motion'] * 0.015 + cost['net2_1st_deriv'] * 0.1 + 0.01 * sum(reg_losses) / len(reg_losses)

    # optimizer
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        t_vars = tf.trainable_variables()
        net1_vars = [var for var in t_vars if 'net1_' in var.name]
        net2_vars = [var for var in t_vars if 'net2_' in var.name]

        net2_reg_vars = [var for var in t_vars if ('net2_' in var.name and 'reg' in var.name)]
        net1_pho_vars = [var for var in t_vars if ('net1_' in var.name and 'pho' in var.name)]

        net1_optim = tf.train.AdamOptimizer(learning_rate).minimize(cost['net1'], var_list=net1_vars)
        net2_optim = tf.train.AdamOptimizer(learning_rate).minimize(cost['net2'], var_list=net2_vars)
        all_optim = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(cost['net2_v_reg'], var_list=net2_reg_vars)

    # Initializing the variables
    init = tf.global_variables_initializer()

    # cv error for avg
    with tf.name_scope('err_avg'):
        sum_val = dict()
        clear_op = dict()
        inc_op = dict()
        avg = dict()
        sum_val['batch'] = tf.Variable(0.)
        clear_op['batch'] = tf.assign(sum_val['batch'], 0.)
        inc_op['batch'] = tf.assign_add(sum_val['batch'], batch_size_placeholder)
        for key in ['net1_land', 'net1_pho', 'net1', 'net1_pho_err', 'net1_regularization', 'net2_v_cls', 'net2_v_reg', 'net2_jali', 'net2', 'net1_motion', 'net2_motion', 'net2_1st_deriv']:
            sum_val[key] = tf.Variable(0.)
            clear_op[key] = tf.assign(sum_val[key], 0.)
            inc_op[key] = tf.assign_add(sum_val[key], cost[key] * batch_size_placeholder)
            avg[key] = tf.divide(sum_val[key], sum_val['batch'])

    tensorboard_op = dict()
    for d_type in ['Train', 'Val']:
        with tf.name_scope(d_type + '_tensorboard'):
            for key in ['net1_land', 'net1_pho', 'net1', 'net1_pho_err', 'net1_regularization', 'net2_v_cls', 'net2_v_reg', 'net2_jali', 'net2', 'net1_motion', 'net2_motion', 'net2_1st_deriv']:
                tf.summary.scalar(d_type + '_' + key, avg[key], collections=[d_type])
            tensorboard_op[d_type] = tf.summary.merge_all(d_type)

    # check_param_num(net1_vars)

    return init, net1_optim, net2_optim, all_optim, x, x_face_id, y_landmark, y_phoneme, y_lipS, y_maya_param, dropout, cost, \
           tensorboard_op, pred, clear_op, inc_op, avg, batch_size_placeholder, phase

    # return init, optimizer, optimizer_landmark, optimizer_phoneme, x, y_landmark, y_phoneme, dropout, cost, error, \
    #       summary_op_train, summary_op_cv, confusion_matrix, pred

import pandas as pd
import numpy as np
from sklearn.metrics import auc
from sklearn.utils.extmath import stable_cumsum
from sklearn.utils.validation import check_consistent_length
from sklearn.metrics import make_scorer
from keras.models import load_model
import os
import random
import datetime
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.keras.metrics import AUC, Precision, Recall, MeanSquaredError
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Input, Dropout, Layer, Concatenate, Flatten, Add, MultiHeadAttention, \
    Bidirectional, GRU, Attention, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers, optimizers
from keras.layers.core import Lambda

try:
    from tensorflow.python.ops.init_ops_v2 import Zeros, Ones, glorot_normal
except ImportError:
    from tensorflow.python.ops.init_ops import Zeros, Ones, glorot_normal_initializer as glorot_normal

from tensorflow.python.keras.layers import Layer, Dropout

try:
    from tensorflow.python.keras.layers import BatchNormalization
except ImportError:
    BatchNormalization = tf.keras.layers.BatchNormalization
from tensorflow.python.keras.regularizers import l2

import tensorflow as tf

try:
    from tensorflow.python.ops.init_ops import Zeros
except ImportError:
    from tensorflow.python.ops.init_ops_v2 import Zeros
from tensorflow.python.keras.layers import Layer, Activation

try:
    from tensorflow.python.keras.layers import BatchNormalization
except ImportError:
    BatchNormalization = tf.keras.layers.BatchNormalization

try:
    unicode
except NameError:
    unicode = str

random.seed(42)
tf.keras.utils.set_random_seed(42)
np.random.seed(42)
os.environ['TF_DETERMINISTIC_OPS'] = '1'


def activation_layer(activation):
    if activation in ("dice", "Dice"):
        act_layer = Dice()
    elif isinstance(activation, (str, unicode)):
        act_layer = Activation(activation)
    elif issubclass(activation, Layer):
        act_layer = activation()
    else:
        raise ValueError(
            "Invalid activation,found %s.You should use a str or a Activation Layer Class." % (activation))
    return act_layer


def reduce_sum(input_tensor,
               axis=None,
               keep_dims=False,
               name=None,
               reduction_indices=None):
    try:
        return tf.reduce_sum(input_tensor,
                             axis=axis,
                             keep_dims=keep_dims,
                             name=name,
                             reduction_indices=reduction_indices)
    except TypeError:
        return tf.reduce_sum(input_tensor,
                             axis=axis,
                             keepdims=keep_dims,
                             name=name)


class DNN(Layer):
    def __init__(self, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False, output_activation=None,
                 seed=1024, **kwargs):
        self.hidden_units = hidden_units
        self.activation = activation
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.use_bn = use_bn
        self.output_activation = output_activation
        self.seed = seed

        super(DNN, self).__init__(**kwargs)

    def build(self, input_shape):
        # if len(self.hidden_units) == 0:
        #     raise ValueError("hidden_units is empty")
        input_size = input_shape[-1]
        hidden_units = [int(input_size)] + list(self.hidden_units)
        self.kernels = [self.add_weight(name='kernel' + str(i),
                                        shape=(
                                            hidden_units[i], hidden_units[i + 1]),
                                        initializer=glorot_normal(
                                            seed=self.seed),
                                        regularizer=l2(self.l2_reg),
                                        trainable=True) for i in range(len(self.hidden_units))]
        self.bias = [self.add_weight(name='bias' + str(i),
                                     shape=(self.hidden_units[i],),
                                     initializer=Zeros(),
                                     trainable=True) for i in range(len(self.hidden_units))]
        if self.use_bn:
            self.bn_layers = [BatchNormalization() for _ in range(len(self.hidden_units))]

        self.dropout_layers = [Dropout(self.dropout_rate, seed=self.seed + i) for i in
                               range(len(self.hidden_units))]

        self.activation_layers = [activation_layer(self.activation) for _ in range(len(self.hidden_units))]

        if self.output_activation:
            self.activation_layers[-1] = activation_layer(self.output_activation)

        super(DNN, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, training=None, **kwargs):

        deep_input = inputs

        for i in range(len(self.hidden_units)):
            fc = tf.nn.bias_add(tf.tensordot(
                deep_input, self.kernels[i], axes=(-1, 0)), self.bias[i])

            if self.use_bn:
                fc = self.bn_layers[i](fc, training=training)
            try:
                fc = self.activation_layers[i](fc, training=training)
            except TypeError as e:  # TypeError: call() got an unexpected keyword argument 'training'
                print("make sure the activation function use training flag properly", e)
                fc = self.activation_layers[i](fc)

            fc = self.dropout_layers[i](fc, training=training)
            deep_input = fc

        return deep_input

    def compute_output_shape(self, input_shape):
        if len(self.hidden_units) > 0:
            shape = input_shape[:-1] + (self.hidden_units[-1],)
        else:
            shape = input_shape

        return tuple(shape)

    def get_config(self, ):
        config = {'activation': self.activation, 'hidden_units': self.hidden_units,
                  'l2_reg': self.l2_reg, 'use_bn': self.use_bn, 'dropout_rate': self.dropout_rate,
                  'output_activation': self.output_activation, 'seed': self.seed}
        base_config = super(DNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def DataRead(data, batch_size):
    features_datas = data[0]
    labels_datas = data[1]
    treat_datas = data[2]
    # print("file_name_li", file_name_li)
    print("treat_datas", treat_datas)
    if len(features) > 1:
        while True:
            num_samples = features_datas.shape[0]
            # yield吐出数据
            for offset in range(0, num_samples, batch_size):
                common_feats_input_data = []
                batch_df = features_datas.iloc[offset:offset + batch_size]
                for col in features:
                    common_feats_input_data.append(np.array(batch_df[col].tolist()))
                #                 common_feats_input_data.append(treat_datas.iloc[offset:offset + batch_size].values)
                mask = treat_datas.iloc[offset:offset + batch_size].values
                label = labels_datas.iloc[offset:offset + batch_size].values
                common_feats_input_data.append(mask)
                yield common_feats_input_data, [label * mask, label * (1 - mask)]
    else:
        raise FileExistsError(f'【{data_path}】file is empty!')


class UpliftModel:
    def __init__(self, embedding_dim, seq_len, batch_size, train_data_size, valid_data_size, train, test):
        self.embedding_dim = embedding_dim
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.train_data_size = train_data_size
        self.valid_data_size = valid_data_size
        self.train = train
        self.test = test

    def mlp_1(self, input_emb, task_type):
        tmp_emb = Dense(64, activation='relu', name=task_type + '_hidden1')(input_emb)
        out_emb = Dense(32, activation='sigmoid', name=task_type + '_out')(tmp_emb)
        return out_emb

    def mtl_module4(self, mlp_inputs, task_names, masks, indicator_emb, gate_emb):
        expert_outs = []
        expert_dnn_hidden_units = (128, 64)
        dnn_dropout = 0
        dnn_activation = 'relu'
        l2_reg_dnn = 0
        dnn_use_bn = False
        num_experts = 3
        for i in range(num_experts):
            expert_network = DNN(expert_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn,
                                 name='expert_' + str(i))(mlp_inputs)
            expert_outs.append(expert_network)

        expert_concat = Lambda(lambda x: tf.stack(x, axis=1))(expert_outs)  # None,num_experts,dim

        mmoe_outs = []
        gate_dnn_hidden_units = ()
        for i in range(2):  # one mmoe layer: nums_tasks = num_gates
            # build gate layers
            gate_input = DNN(gate_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn,
                             name='gate_' + task_names[i])(gate_emb)
            gate_out = Dense(num_experts, use_bias=False, activation='softmax',
                             name='gate_softmax_' + task_names[i])(gate_input)
            gate_out = Lambda(lambda x: tf.expand_dims(x, axis=-1))(gate_out)

            # gate multiply the expert
            gate_mul_expert = Lambda(lambda x: reduce_sum(x[0] * x[1], axis=1, keep_dims=False),
                                     name='gate_mul_expert_' + task_names[i])([expert_concat, gate_out])
            mmoe_outs.append(gate_mul_expert)

        task_outs = []
        tower_dnn_hidden_units = (32,)
        for task_name, mmoe_out, mask in zip(task_names, mmoe_outs, masks):
            # build tower layer
            tower_output_tmp = DNN(tower_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn,
                                   name='tower_' + task_name)(mmoe_out)
            tower_output = indicator_emb * tower_output_tmp
            output = Dense(1, activation='softplus', name=f'{task_name}_task_out')(tower_output)

            #         logit = Dense(1, use_bias=False)(tower_output)
            task_outs.append(tf.keras.layers.multiply([output, mask], name=f'{task_name}_task'))
        return task_outs

    def build_model(self):
        uplift_feats_sparse_inputs_li = []
        uplift_feats_sparse_embeddings_li = []
        for idx, feat in enumerate(features):
            input_layer = Input(shape=(1,), name=feat + '_input')
            dim = 100
            denominator = max(1e-8, denominators[idx])
            input_t = input_layer / denominator * dim
            input_int = tf.minimum(tf.maximum(tf.cast(input_t, tf.int32), 0), dim)

            embedding = layers.Embedding(input_dim=dim + 1, output_dim=self.embedding_dim, mask_zero=True,
                                         name=feat + '_embedding')(input_int)
            uplift_feats_sparse_inputs_li.append(input_layer)
            uplift_feats_sparse_embeddings_li.append(embedding)
        tf.print("==*== uplift_feats_sparse[0] embedding shape = ", uplift_feats_sparse_embeddings_li[0].shape)

        treat_mask = Input(shape=(1,), name='exp_name_input')
        treat_embedding = layers.Embedding(input_dim=2, output_dim=self.embedding_dim * 4, mask_zero=True,
                                           name='treat_embedding')(treat_mask)
        uplift_feats_sparse_inputs_li.append(treat_mask)
        concat_embedding = Concatenate(axis=1)(uplift_feats_sparse_embeddings_li)
        tf.print('==*== concat_embedding shape = ', concat_embedding.shape)

        mlp_inputs = Flatten()(concat_embedding)

        indicator_emb = self.mlp_1(Flatten()(treat_embedding), 'indicator_emb')
        guide_emb = self.mlp_1(Flatten()(treat_embedding), 'guide_emb')
        tf.print('==*== mlp_inputs shape = ', mlp_inputs.shape)

        task_outputs = self.mtl_module4(mlp_inputs, ["treat", "base"],
                                        [treat_mask, tf.ones_like(treat_mask) - treat_mask], indicator_emb, guide_emb)

        tf.print('==*== task_1_outputs shape= {} '.format(task_outputs[0].shape))
        model = Model(inputs=uplift_feats_sparse_inputs_li, outputs=task_outputs)
        model.summary()
        return model

    def get_loss(self, y_true, y_pred):
        #       loss = tf.compat.v1.losses.log_loss(y_pred, y_true)
        weights = tf.reshape(self.treat_mask, (-1, 1))
        mse = tf.square(y_true - y_pred)
        weighted_loss = mse * weights
        return tf.reduce_mean(weighted_loss)

    def get_loss2(self, y_true, y_pred):
        #       loss = tf.compat.v1.losses.log_loss(y_pred, y_true)
        treat_mask2 = (tf.ones_like(self.treat_mask) - self.treat_mask)
        weights = tf.reshape(treat_mask2, (-1, 1))
        mse = tf.square(y_true - y_pred)
        weighted_loss = mse * weights
        return tf.reduce_mean(weighted_loss)

    def get_loss3(self, y_true, y_pred):
        #       loss = tf.compat.v1.losses.log_loss(y_pred, y_true)
        loss_layer = WeightedLossLayer()
        weights = tf.reshape(self.treat_mask, (-1, 1))
        mse = tf.square(y_true - y_pred)
        weighted_loss = mse * weights
        return tf.reduce_mean(weighted_loss)

    def get_data(self):
        train_data = DataRead(self.train, batch_size=self.batch_size)
        valid_data = DataRead(self.test, batch_size=self.batch_size)
        return train_data, valid_data

    def model_conf(self):
        train_date = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        checkpoint_path = './gamora_uplift_model_v1_' + train_date + '_' + str(embedding_dim) + '_' + str(batch_size)
        import os
        if os.path.isdir(checkpoint_path) is not True:
            os.mkdir(checkpoint_path)
        print('--- train date = ', train_date)

        # 设置损失权重
        self.loss_weights = {
            'treat_task': 1.0,
            'base_task': 1.0
        }

        self.metrics = {
            'treat_task': [MeanSquaredError()],
            'base_task': [MeanSquaredError()]
        }

        self.checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_path + '/epoch_{epoch:02d}.h5',  # 模型文件保存路径
            save_weights_only=True,  # 是否只保存模型权重
            save_freq='epoch'  # 保存频率
        )

        self.reduce_lr_callback = ReduceLROnPlateau(
            monitor='val_loss',  # 监控的指标
            factor=0.6,  # 学习率减少的因子，new_lr = lr * factor
            patience=2,  # 没有改进的 epoch 数
            min_lr=1e-6,  # 学习率的下限
            verbose=1  # 输出学习率变化信息
        )

    def model_train(self):
        gpus = tf.config.experimental.list_physical_devices('GPU')

        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            #                 tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
            #                 tf.config.experimental.set_memory_growth(gpus[1], True)
            except RuntimeError as e:
                print(e)

        strategy = tf.distribute.MirroredStrategy()
        train_data, valid_data = self.get_data()
        valid_data = DataRead(self.test, batch_size=self.batch_size)
        with strategy.scope():
            self.model_conf()
            model = self.build_model()
            optimizer = optimizers.Adam(lr=0.001)
            #             # 编译模型
            model.compile(optimizer=optimizer, loss={
                'treat_task': 'huber',
                'base_task': 'huber'
            }, loss_weights=self.loss_weights, metrics=self.metrics)
            steps_per_epoch_train = self.train_data_size // self.batch_size
            steps_per_epoch_valid = self.valid_data_size // self.batch_size
            print(train_data)
            # history = model.fit(
            #     valid_data
            #     ,epochs=1
            #     ,steps_per_epoch=steps_per_epoch_valid
            #     ,validation_data = valid_data
            #     ,validation_steps=steps_per_epoch_valid
            #     ,verbose=1
            #     ,shuffle=True
            #     ,callbacks=[self.checkpoint_callback,self.reduce_lr_callback]
            # )

        return model


tmp_test_path = './criteo/criteo-test.csv'
if os.path.exists(tmp_test_path):
    print("exist")
    sampled_test_data = pd.read_csv(tmp_test_path)

tmp_val_path = './criteo/criteo-val.csv'
if os.path.exists(tmp_val_path):
    print("exist")
    sampled_val_data = pd.read_csv(tmp_val_path)

tmp_train_path = './criteo/criteo-train.csv'
if os.path.exists(tmp_train_path):
    print("exist")
    sampled_train_data = pd.read_csv(tmp_train_path)

# df_all = pd.concat([sampled_train_data, sampled_test_data])
df_train = sampled_train_data
df_val = sampled_val_data
df_test = sampled_test_data

features = []
for i in range(12):
    features.append('f' + str(i))

x_train, y_train, treatment_train = df_train[features], df_train['visit'], df_train['treatment']
x_val, y_val, treatment_val = df_val[features], df_val['visit'], df_val['treatment']
x_test, y_test, treatment_test = df_test[features], df_test['visit'], df_test['treatment']

denominators = []
for fea in features:
    denominators.append(df_train[fea].max())


def qini_curve(y_true, uplift, treatment):
    """Compute Qini curve.

    For computing the area under the Qini Curve, see :func:`.qini_auc_score`.

    Args:
        y_true (1d array-like): Correct (true) binary target values.
        uplift (1d array-like): Predicted uplift, as returned by a model.
        treatment (1d array-like): Treatment labels.

    Returns:
        array (shape = [>2]), array (shape = [>2]): Points on a curve.

    See also:
        :func:`.uplift_curve`: Compute the area under the Qini curve.

        :func:`.perfect_qini_curve`: Compute the perfect Qini curve.

        :func:`.plot_qini_curves`: Plot Qini curves from predictions..

        :func:`.uplift_curve`: Compute Uplift curve.

    References:
        Nicholas J Radcliffe. (2007). Using control groups to target on predicted lift:
        Building and assessing uplift model. Direct Marketing Analytics Journal, (3):14–21, 2007.

        Devriendt, F., Guns, T., & Verbeke, W. (2020). Learning to rank for uplift modeling. ArXiv, abs/2002.05897.
    """

    check_consistent_length(y_true, uplift, treatment)
    #     check_is_binary(treatment)
    #     check_is_binary(y_true)
    y_true, uplift, treatment = np.array(y_true), np.array(uplift), np.array(treatment)

    desc_score_indices = np.argsort(uplift, kind="mergesort")[::-1]

    y_true = y_true[desc_score_indices]
    treatment = treatment[desc_score_indices]
    uplift = uplift[desc_score_indices]

    y_true_ctrl, y_true_trmnt = y_true.copy(), y_true.copy()

    y_true_ctrl[treatment == 1] = 0
    y_true_trmnt[treatment == 0] = 0

    distinct_value_indices = np.where(np.diff(uplift))[0]
    threshold_indices = np.r_[distinct_value_indices, uplift.size - 1]

    num_trmnt = stable_cumsum(treatment)[threshold_indices]
    y_trmnt = stable_cumsum(y_true_trmnt)[threshold_indices]

    num_all = threshold_indices + 1

    num_ctrl = num_all - num_trmnt
    y_ctrl = stable_cumsum(y_true_ctrl)[threshold_indices]

    curve_values = y_trmnt - y_ctrl * np.divide(num_trmnt, num_ctrl, out=np.zeros_like(num_trmnt), where=num_ctrl != 0)
    if num_all.size == 0 or curve_values[0] != 0 or num_all[0] != 0:
        # Add an extra threshold position if necessary
        # to make sure that the curve starts at (0, 0)
        num_all = np.r_[0, num_all]
        curve_values = np.r_[0, curve_values]

    return num_all, curve_values


def uplift_curve(y_true, uplift, treatment):
    """Compute Uplift curve.

    For computing the area under the Uplift Curve, see :func:`.uplift_auc_score`.

    Args:
        y_true (1d array-like): Correct (true) binary target values.
        uplift (1d array-like): Predicted uplift, as returned by a model.
        treatment (1d array-like): Treatment labels.

    Returns:
        array (shape = [>2]), array (shape = [>2]): Points on a curve.

    See also:
        :func:`.uplift_auc_score`: Compute normalized Area Under the Uplift curve from prediction scores.

        :func:`.perfect_uplift_curve`: Compute the perfect Uplift curve.

        :func:`.plot_uplift_curve`: Plot Uplift curves from predictions.

        :func:`.qini_curve`: Compute Qini curve.

    References:
        Devriendt, F., Guns, T., & Verbeke, W. (2020). Learning to rank for uplift modeling. ArXiv, abs/2002.05897.
    """

    check_consistent_length(y_true, uplift, treatment)
    #     check_is_binary(treatment)
    #     check_is_binary(y_true)

    y_true, uplift, treatment = np.array(y_true), np.array(uplift), np.array(treatment)

    desc_score_indices = np.argsort(uplift, kind="mergesort")[::-1]
    y_true, uplift, treatment = y_true[desc_score_indices], uplift[desc_score_indices], treatment[desc_score_indices]

    y_true_ctrl, y_true_trmnt = y_true.copy(), y_true.copy()

    y_true_ctrl[treatment == 1] = 0
    y_true_trmnt[treatment == 0] = 0

    distinct_value_indices = np.where(np.diff(uplift))[0]
    threshold_indices = np.r_[distinct_value_indices, uplift.size - 1]

    num_trmnt = stable_cumsum(treatment)[threshold_indices]
    y_trmnt = stable_cumsum(y_true_trmnt)[threshold_indices]

    num_all = threshold_indices + 1

    num_ctrl = num_all - num_trmnt
    y_ctrl = stable_cumsum(y_true_ctrl)[threshold_indices]

    curve_values = (np.divide(y_trmnt, num_trmnt, out=np.zeros_like(y_trmnt), where=num_trmnt != 0) -
                    np.divide(y_ctrl, num_ctrl, out=np.zeros_like(y_ctrl), where=num_ctrl != 0)) * num_all

    if num_all.size == 0 or curve_values[0] != 0 or num_all[0] != 0:
        # Add an extra threshold position if necessary
        # to make sure that the curve starts at (0, 0)
        num_all = np.r_[0, num_all]
        curve_values = np.r_[0, curve_values]

    return num_all, curve_values


def perfect_uplift_curve(y_true, treatment):
    """Compute the perfect (optimum) Uplift curve.

    This is a function, given points on a curve.  For computing the
    area under the Uplift Curve, see :func:`.uplift_auc_score`.

    Args:
        y_true (1d array-like): Correct (true) binary target values.
        treatment (1d array-like): Treatment labels.

    Returns:
        array (shape = [>2]), array (shape = [>2]): Points on a curve.

    See also:
        :func:`.uplift_curve`: Compute the area under the Qini curve.

        :func:`.uplift_auc_score`: Compute normalized Area Under the Uplift curve from prediction scores.

        :func:`.plot_uplift_curve`: Plot Uplift curves from predictions.
    """

    check_consistent_length(y_true, treatment)
    #     check_is_binary(treatment)
    #     check_is_binary(y_true)
    y_true, treatment = np.array(y_true), np.array(treatment)

    cr_num = np.sum((y_true == 1) & (treatment == 0))  # Control Responders
    tn_num = np.sum((y_true == 0) & (treatment == 1))  # Treated Non-Responders

    # express an ideal uplift curve through y_true and treatment
    summand = y_true if cr_num > tn_num else treatment
    perfect_uplift = 2 * (y_true == treatment) + summand

    return uplift_curve(y_true, perfect_uplift, treatment)


def perfect_qini_curve(y_true, treatment, negative_effect=True):
    """Compute the perfect (optimum) Qini curve.

    For computing the area under the Qini Curve, see :func:`.qini_auc_score`.

    Args:
        y_true (1d array-like): Correct (true) binary target values.
        treatment (1d array-like): Treatment labels.
        negative_effect (bool): If True, optimum Qini Curve contains the negative effects
            (negative uplift because of campaign). Otherwise, optimum Qini Curve will not
            contain the negative effects.
    Returns:
        array (shape = [>2]), array (shape = [>2]): Points on a curve.

    See also:
        :func:`.qini_curve`: Compute Qini curve.

        :func:`.qini_auc_score`: Compute the area under the Qini curve.

        :func:`.plot_qini_curves`: Plot Qini curves from predictions..
    """

    check_consistent_length(y_true, treatment)
    #     check_is_binary(treatment)
    #     check_is_binary(y_true)
    n_samples = len(y_true)

    y_true, treatment = np.array(y_true), np.array(treatment)

    if not isinstance(negative_effect, bool):
        raise TypeError(f'Negative_effects flag should be bool, got: {type(negative_effect)}')

    # express an ideal uplift curve through y_true and treatment
    if negative_effect:
        x_perfect, y_perfect = qini_curve(
            y_true, y_true * treatment - y_true * (1 - treatment), treatment
        )
    else:
        ratio_random = (y_true[treatment == 1].sum() - len(y_true[treatment == 1]) *
                        y_true[treatment == 0].sum() / len(y_true[treatment == 0]))

        x_perfect, y_perfect = np.array([0, ratio_random, n_samples]), np.array([0, ratio_random, ratio_random])

    return x_perfect, y_perfect


def uplift_auc_score1(y_true, uplift, treatment):
    """Compute normalized Area Under the Uplift Curve from prediction scores.

    By computing the area under the Uplift curve, the curve information is summarized in one number.
    For binary outcomes the ratio of the actual uplift gains curve above the diagonal to that of
    the optimum Uplift Curve.

    Args:
        y_true (1d array-like): Correct (true) binary target values.
        uplift (1d array-like): Predicted uplift, as returned by a model.
        treatment (1d array-like): Treatment labels.

    Returns:
        float: Area Under the Uplift Curve.

    See also:
        :func:`.uplift_curve`: Compute Uplift curve.

        :func:`.perfect_uplift_curve`: Compute the perfect (optimum) Uplift curve.

        :func:`.plot_uplift_curve`: Plot Uplift curves from predictions.

        :func:`.qini_auc_score`: Compute normalized Area Under the Qini Curve from prediction scores.
    """

    check_consistent_length(y_true, uplift, treatment)
    #     check_is_binary(treatment)
    #     check_is_binary(y_true)
    y_true, uplift, treatment = np.array(y_true), np.array(uplift), np.array(treatment)

    x_actual, y_actual = uplift_curve(y_true, uplift, treatment)
    x_perfect, y_perfect = perfect_uplift_curve(y_true, treatment)
    x_baseline, y_baseline = np.array([0, x_perfect[-1]]), np.array([0, y_perfect[-1]])

    auc_score_baseline = auc(x_baseline, y_baseline)
    auc_score_perfect = auc(x_perfect, y_perfect) - auc_score_baseline
    auc_score_actual = auc(x_actual, y_actual) - auc_score_baseline

    return auc_score_actual / auc_score_perfect


# qini
def qini_auc_score1(y_true, uplift, treatment, negative_effect=True):
    """Compute normalized Area Under the Qini curve (aka Qini coefficient) from prediction scores.

    By computing the area under the Qini curve, the curve information is summarized in one number.
    For binary outcomes the ratio of the actual uplift gains curve above the diagonal to that of
    the optimum Qini curve.

    Args:
        y_true (1d array-like): Correct (true) binary target values.
        uplift (1d array-like): Predicted uplift, as returned by a model.
        treatment (1d array-like): Treatment labels.
        negative_effect (bool): If True, optimum Qini Curve contains the negative effects
            (negative uplift because of campaign). Otherwise, optimum Qini Curve will not contain the negative effects.

            .. versionadded:: 0.2.0

    Returns:
        float: Qini coefficient.

    See also:
        :func:`.qini_curve`: Compute Qini curve.

        :func:`.perfect_qini_curve`: Compute the perfect (optimum) Qini curve.

        :func:`.plot_qini_curves`: Plot Qini curves from predictions..

        :func:`.uplift_auc_score`: Compute normalized Area Under the Uplift curve from prediction scores.

    References:
        Nicholas J Radcliffe. (2007). Using control groups to target on predicted lift:
        Building and assessing uplift model. Direct Marketing Analytics Journal, (3):14–21, 2007.
    """

    # TODO: Add Continuous Outcomes
    check_consistent_length(y_true, uplift, treatment)
    #     check_is_binary(treatment)
    #     check_is_binary(y_true)
    y_true, uplift, treatment = np.array(y_true), np.array(uplift), np.array(treatment)

    x_actual, y_actual = qini_curve(y_true, uplift, treatment)
    x_perfect, y_perfect = perfect_qini_curve(y_true, treatment, negative_effect)
    x_baseline, y_baseline = np.array([0, x_perfect[-1]]), np.array([0, y_perfect[-1]])

    auc_score_baseline = auc(x_baseline, y_baseline)
    auc_score_perfect = auc(x_perfect, y_perfect) - auc_score_baseline
    auc_score_actual = auc(x_actual, y_actual) - auc_score_baseline
    print("qini_auc_score_perfect", auc_score_perfect)
    print("qini_auc_score_baseline", auc_score_baseline)
    print("qini_auc_score_actual", auc_score_actual)

    return auc_score_actual / auc_score_perfect


def uplift_at_k1(y_true, uplift, treatment, strategy, k=0.3):
    """Compute uplift at first k observations by uplift of the total sample.

    Args:
        y_true (1d array-like): Correct (true) binary target values.
        uplift (1d array-like): Predicted uplift, as returned by a model.
        treatment (1d array-like): Treatment labels.
        k (float or int): If float, should be between 0.0 and 1.0 and represent the proportion of the dataset
            to include in the computation of uplift. If int, represents the absolute number of samples.
        strategy (string, ['overall', 'by_group']): Determines the calculating strategy.

            * ``'overall'``:
                The first step is taking the first k observations of all test data ordered by uplift prediction
                (overall both groups - control and treatment) and conversions in treatment and control groups
                calculated only on them. Then the difference between these conversions is calculated.

            * ``'by_group'``:
                Separately calculates conversions in top k observations in each group (control and treatment)
                sorted by uplift predictions. Then the difference between these conversions is calculated



    .. versionchanged:: 0.1.0

        * Add supporting absolute values for ``k`` parameter
        * Add parameter ``strategy``

    Returns:
        float: Uplift score at first k observations of the total sample.

    See also:
        :func:`.uplift_auc_score`: Compute normalized Area Under the Uplift curve from prediction scores.

        :func:`.qini_auc_score`: Compute normalized Area Under the Qini Curve from prediction scores.
    """

    # TODO: checker all groups is not empty
    check_consistent_length(y_true, uplift, treatment)
    # check_is_binary(treatment)
    # check_is_binary(y_true)
    y_true, uplift, treatment = np.array(y_true), np.array(uplift), np.array(treatment)

    strategy_methods = ['overall', 'by_group']
    if strategy not in strategy_methods:
        raise ValueError(f'Uplift score supports only calculating methods in {strategy_methods},'
                         f' got {strategy}.'
                         )

    n_samples = len(y_true)
    order = np.argsort(uplift, kind='mergesort')[::-1]
    _, treatment_counts = np.unique(treatment, return_counts=True)
    n_samples_ctrl = treatment_counts[0]
    n_samples_trmnt = treatment_counts[1]

    k_type = np.asarray(k).dtype.kind

    if (k_type == 'i' and (k >= n_samples or k <= 0)
            or k_type == 'f' and (k <= 0 or k >= 1)):
        raise ValueError(f'k={k} should be either positive and smaller'
                         f' than the number of samples {n_samples} or a float in the '
                         f'(0, 1) range')

    if k_type not in ('i', 'f'):
        raise ValueError(f'Invalid value for k: {k_type}')

    if strategy == 'overall':
        if k_type == 'f':
            n_size = int(n_samples * k)
        else:
            n_size = k

        # ToDo: _checker_ there are observations among two groups among first k
        score_ctrl = y_true[order][:n_size][treatment[order][:n_size] == 0].mean()
        score_trmnt = y_true[order][:n_size][treatment[order][:n_size] == 1].mean()

    else:  # strategy == 'by_group':
        if k_type == 'f':
            n_ctrl = int((treatment == 0).sum() * k)
            n_trmnt = int((treatment == 1).sum() * k)

        else:
            n_ctrl = k
            n_trmnt = k

        if n_ctrl > n_samples_ctrl:
            raise ValueError(f'With k={k}, the number of the first k observations'
                             ' bigger than the number of samples'
                             f'in the control group: {n_samples_ctrl}'
                             )
        if n_trmnt > n_samples_trmnt:
            raise ValueError(f'With k={k}, the number of the first k observations'
                             ' bigger than the number of samples'
                             f'in the treatment group: {n_samples_ctrl}'
                             )

        score_ctrl = y_true[order][treatment[order] == 0][:n_ctrl].mean()
        score_trmnt = y_true[order][treatment[order] == 1][:n_trmnt].mean()

    return score_trmnt - score_ctrl


def get_loss_pos_neg(y_true, y_pred):
    loss = tf.compat.v1.losses.mean_squared_error(y_pred, y_true)
    return loss


def get_loss_shared(y_true, y_pred):
    loss = tf.compat.v1.losses.mean_squared_error(y_pred, y_true)
    return loss


embedding_dim = 32
seq_len = 10
batch_size = 4096
train_data_size = x_train.shape[0]
valid_data_size = x_val.shape[0]
Uplift = UpliftModel(embedding_dim=embedding_dim, seq_len=seq_len, batch_size=batch_size,
                     train_data_size=train_data_size, valid_data_size=valid_data_size,
                     train=[x_train, y_train, treatment_train], test=[x_val, y_val, treatment_val])
uplift_model = Uplift.model_train()

uplift_model.load_weights('./epoch_criteo.h5')

base_test_data = []
treat_test_data = []
control_test_data = []

for col in features:
    base_test_data.append(x_test[col].to_numpy())

treat_test_data += base_test_data + [np.ones(x_test.shape[0], )]
control_test_data += base_test_data + [np.zeros(x_test.shape[0], )]
res_1, _ = uplift_model.predict(treat_test_data)
_, res_0 = uplift_model.predict(control_test_data)
print((res_1 - res_0).shape)
uplift_auc = uplift_auc_score1(y_true=y_test, uplift=np.array(res_1 - res_0).reshape(1, -1)[0],
                               treatment=treatment_test)
qini_auc = qini_auc_score1(y_true=y_test, uplift=np.array(res_1 - res_0).reshape(1, -1)[0],
                           treatment=treatment_test)
uplift_at_k_auc = uplift_at_k1(y_true=y_test, uplift=np.array(res_1 - res_0).reshape(1, -1)[0],
                               treatment=treatment_test, strategy='overall', k=0.3)
print(uplift_auc)
print(qini_auc)
print(uplift_at_k_auc)

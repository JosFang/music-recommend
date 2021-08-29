import tensorflow as tf
import math
import numpy as np
import time
import sys
from data import *
from utils import *

LOSS = []
HIT_RATE = []
MRR = []

class music_recommend_model:
    def __init__(self, args, music_num, user_musics_dic, paths):
        self.dropout_train = args.dropout
        self.n_layer = args.n_layer
        self.music_num = music_num
        self.hidden_dim = args.hidden_dim
        self.sequence_length = args.sequence_length
        self.num_attention_heads = args.num_attention_heads
        self.intermediate_size = args.intermediate_size
        self.lr = args.lr
        self.epoch_num = args.epoch_num
        self.batch_size = args.batch_size
        self.user_musics_dic = user_musics_dic
        self.logger = get_logger(paths['log_path'])
        self.model_path = paths['model_path']
        self.neg_num = args.neg_num
        self.top_k = args.top_k
        self.batch_size_test = args.batch_size_test

    def build_graph(self):
        self.add_placeholders()
        self.lookup_layer_op()
        self.transformer()
        self.loss_op()
        self.trainstep_op()

    def add_placeholders(self):
        self.music_id = tf.placeholder(tf.int32, shape=[None, self.sequence_length],
                                       name="music_id")
        self.lengths = tf.placeholder(tf.int32, shape=[None],
                                         name="lengths")
        self.precursor = tf.placeholder(tf.int32, shape=[None, None], name="precursor")
        self.pre_length = tf.placeholder(tf.int32, shape=[None], name="pre_length")
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.is_train = tf.placeholder(dtype=tf.int32, shape=[], name="is_train")

    def lookup_layer_op(self):
        with tf.variable_scope("input_layer"):
            self.music_embedding_all = tf.get_variable(name='music_embedding_all',
                                                   shape=[self.music_num, self.hidden_dim],
                                                   initializer=tf.random_uniform_initializer(),
                                                   dtype=tf.float32)
            self.music_embedding = tf.nn.embedding_lookup(params=self.music_embedding_all,
                                                                     ids=self.music_id,
                                                                     name="music_embedding")
            # [batch_size, sequence_length, hidden_dim]
            self.music_embedding = tf.nn.dropout(self.music_embedding, self.dropout)

            self.position_embeddings = tf.get_variable(
                name="position_embeddings",
                shape=[self.sequence_length, self.hidden_dim],
                initializer=tf.random_uniform_initializer())
            position_embeddings = tf.reshape(self.position_embeddings,
                                             [1, self.sequence_length, self.hidden_dim])
            self.output_embedding = self.music_embedding + position_embeddings
            self.output_embedding = self.layer_norm(self.output_embedding)
            # [batch_size, sequence_length, hidden_dim]
            self.output_embedding = tf.nn.dropout(self.output_embedding, self.dropout)

    def transformer(self):
        def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                                 seq_length, width):
            output_tensor = tf.reshape(
                input_tensor, [batch_size, seq_length, num_attention_heads, width])

            output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
            return output_tensor

        def get_shape_list(tensor, expected_rank=None, name=None):
            """Returns a list of the shape of tensor, preferring static dimensions.

            Args:
              tensor: A tf.Tensor object to find the shape of.
              expected_rank: (optional) int. The expected rank of `tensor`. If this is
                specified and the `tensor` has a different rank, and exception will be
                thrown.
              name: Optional name of the tensor for the error message.

            Returns:
              A list of dimensions of the shape of tensor. All static dimensions will
              be returned as python integers, and dynamic dimensions will be returned
              as tf.Tensor scalars.
            """
            if name is None:
                name = tensor.name


            shape = tensor.shape.as_list()

            non_static_indexes = []
            for (index, dim) in enumerate(shape):
                if dim is None:
                    non_static_indexes.append(index)

            if not non_static_indexes:
                return shape

            dyn_shape = tf.shape(tensor)
            for index in non_static_indexes:
                shape[index] = dyn_shape[index]
            return shape

        def create_initializer(initializer_range=0.02):
            """Creates a `truncated_normal_initializer` with the given range."""
            return tf.truncated_normal_initializer(stddev=initializer_range)

        def gelu(x):
            """Gaussian Error Linear Unit.

            This is a smoother version of the RELU.
            Original paper: https://arxiv.org/abs/1606.08415
            Args:
              x: float Tensor to perform activation.

            Returns:
              `x` with the GELU activation applied.
            """
            cdf = 0.5 * (1.0 + tf.tanh(
                (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
            return x * cdf

        def reshape_from_matrix(output_tensor, orig_shape_list):
            """Reshapes a rank 2 tensor back to its original rank >= 2 tensor."""
            if len(orig_shape_list) == 2:
                return output_tensor

            output_shape = get_shape_list(output_tensor)

            orig_dims = orig_shape_list[0:-1]
            width = output_shape[-1]

            return tf.reshape(output_tensor, orig_dims + [width])


        with tf.variable_scope("encoder"):
            # [batch_size, pre_length, hidden_dim]
            # print(self.precursor)
            # self.precursor = tf.Print(self.precursor, [self.precursor])
            precursor = tf.nn.embedding_lookup(params=self.music_embedding_all,
                                               ids=self.precursor,
                                               name="precursor")
            # [batch_size, pre_length, 1]
            tran_mask = tf.to_float(tf.expand_dims(tf.sequence_mask(self.pre_length), -1))
            # [batch_size, hidden_dim]
            tran_fist = tf.divide(tf.reduce_sum(precursor * tran_mask, 1),
                                  tf.reshape(tf.to_float(self.pre_length), [-1, 1]))
            tran_inputs = tf.concat([tf.expand_dims(tran_fist, 1), self.output_embedding], 1)[:, :-1]

            if self.hidden_dim % self.num_attention_heads != 0:
                raise ValueError(
                    "The hidden size (%d) is not a multiple of the number of attention "
                    "heads (%d)" % (self.hidden_dim, self.num_attention_heads))
            attention_head_size = int(self.hidden_dim / self.num_attention_heads)
            input_shape = get_shape_list(tran_inputs)

            prev_output = self.reshape_to_matrix(tran_inputs)
            all_layer_outputs = []
            for layer_idx in range(self.n_layer):

                with tf.variable_scope("layer_%d" % layer_idx):
                    layer_input = prev_output
                    with tf.variable_scope("attention"):
                        from_shape = get_shape_list(tran_inputs)
                        batch_size = from_shape[0]
                        seq_length = from_shape[1]
                        from_tensor_2d = self.reshape_to_matrix(layer_input)

                        query_layer = tf.layers.dense(
                            from_tensor_2d,
                            self.num_attention_heads * attention_head_size,
                            activation=None,
                            name="query",
                            kernel_initializer=create_initializer())

                        # `key_layer` = [B*F, N*H]
                        key_layer = tf.layers.dense(
                            from_tensor_2d,
                            self.num_attention_heads * attention_head_size,
                            activation=None,
                            name="key",
                            kernel_initializer=create_initializer())

                        # `value_layer` = [B*F, N*H]
                        value_layer = tf.layers.dense(
                            from_tensor_2d,
                            self.num_attention_heads * attention_head_size,
                            activation=None,
                            name="value",
                            kernel_initializer=create_initializer())
                        # `key_layer` = [B, N, F, H]
                        key_layer = transpose_for_scores(key_layer, batch_size,
                                                         self.num_attention_heads,
                                                         self.sequence_length,
                                                         attention_head_size)
                        # `query_layer` = [B, N, F, H]
                        query_layer = transpose_for_scores(query_layer, batch_size,
                                                         self.num_attention_heads,
                                                         self.sequence_length,
                                                         attention_head_size)
                        # `attention_scores` = [B, N, F, F]
                        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
                        attention_scores = tf.multiply(attention_scores,
                                                       1.0 / math.sqrt(float(attention_head_size)))

                        # `attention_mask` = [B, 1, F, F]
                        # attention_mask_expand = tf.expand_dims(attention_mask, axis=[1])
                        # 下三角矩阵
                        mask_tri = tf.matrix_band_part(tf.ones([self.sequence_length,
                                                         self.sequence_length]), -1, 0)
                        mask_tri = tf.reshape(mask_tri, [1, 1, self.sequence_length,
                                                         self.sequence_length])
                        attention_scores = attention_scores * mask_tri + -1e9 * (1 - mask_tri)
                        attention_probs = tf.nn.softmax(attention_scores)
                        attention_probs = tf.nn.dropout(attention_probs, self.dropout)
                        value_layer = tf.reshape(
                            value_layer,
                            [batch_size, self.sequence_length, self.num_attention_heads,
                             attention_head_size])
                        # `value_layer` = [B, N, T, H]
                        value_layer = tf.transpose(value_layer, [0, 2, 1, 3])
                        # `context_layer` = [B, N, F, H]
                        context_layer = tf.matmul(attention_probs, value_layer)
                        # `context_layer` = [B, F, N, H]
                        context_layer = tf.transpose(context_layer, [0, 2, 1, 3])
                        context_layer = tf.reshape(
                            context_layer,
                            [batch_size * self.sequence_length,
                             self.num_attention_heads * attention_head_size])
                        # [b_s * s_l, hidden_dim]
                        attention_output = tf.concat(context_layer, axis=-1)
                        with tf.variable_scope("output"):
                            attention_output = tf.layers.dense(
                                attention_output,
                                self.hidden_dim,
                                kernel_initializer=create_initializer())
                            attention_output = tf.nn.dropout(attention_output, self.dropout)
                            attention_output = tf.contrib.layers.layer_norm(
                            inputs=attention_output + layer_input,
                                begin_norm_axis=-1, begin_params_axis=-1)

                    with tf.variable_scope("intermediate"):
                        intermediate_output = tf.layers.dense(
                            attention_output,
                            self.intermediate_size,
                            activation=gelu,
                            kernel_initializer=create_initializer())
                        with tf.variable_scope("output"):
                            layer_output = tf.layers.dense(
                                intermediate_output,
                                self.hidden_dim,
                                kernel_initializer=create_initializer())

                        layer_output = tf.nn.dropout(layer_output, self.dropout)
                        layer_output = tf.contrib.layers.layer_norm(layer_output + attention_output)
                        prev_output = layer_output
                        all_layer_outputs.append(layer_output)
            # [batch_size, sequence_length, hidden_dim]
            self.transformer_out = reshape_from_matrix(prev_output, input_shape)

    def loss_op(self):
        with tf.variable_scope("loss"):
            # [batch_size * seq_length, 1]
            neg_labels = tf.reshape(self.music_id, [-1, 1])
            # [batch_size * seq_length, hidden_dim]
            neg_inputs = tf.reshape(self.transformer_out, [-1, self.hidden_dim])

            nce_weights = tf.get_variable(name='nce_weights',
                                          initializer=tf.truncated_normal([self.music_num,
                                                                           self.hidden_dim],
                                          stddev=1.0 / math.sqrt(self.hidden_dim)))
            nce_biases = tf.get_variable(name='nce_biases', initializer=tf.zeros([self.music_num]))
            mask = tf.sequence_mask(self.lengths)

            # if tf.equal(self.is_train, 1) is not None:
            #     print("房雨帆")
                # 没有mask [batch_size * seq_length]
            loss = tf.nn.nce_loss(weights=nce_weights,
                                  biases=nce_biases,
                                  labels=neg_labels,
                                  inputs=neg_inputs,
                                  num_sampled=self.neg_num,
                                  num_classes=self.music_num,
                                  remove_accidental_hits=True)
            loss = tf.reshape(loss, [-1, self.sequence_length])
            self.loss = tf.reduce_mean(tf.boolean_mask(loss, mask))

            # else:
            #     print("孙香")

            logits = tf.matmul(neg_inputs, tf.transpose(nce_weights))
            # [batch_size*seq_length, music_num]
            self.logits = tf.nn.bias_add(logits, nce_biases)

            # 矩阵分块相乘，处理程序瓶颈
            # [hidden_dim, music_num]
            # nce_weights_t = tf.transpose(nce_weights)
            # part_len = int(self.music_num // 32)
            # part_value = []
            # part_index = []
            # for i in range(31):
            #     part = nce_weights_t[:, i*part_len:(i+1)*part_len]
            #     res = tf.nn.bias_add(tf.matmul(neg_inputs, part),
            #                          nce_biases[i*part_len:(i+1)*part_len])
            #     res_k = tf.nn.top_k(res, self.top_k)
            #     part_index.append(res_k[1])
            #     part_value.append(res_k[0])
            # # if self.music_num % 32 > 0:
            # part = nce_weights_t[:, 31*part_len:]
            # res = tf.nn.bias_add(tf.matmul(neg_inputs, part),
            #                      nce_biases[31*part_len:])
            # res_k = tf.nn.top_k(res, self.top_k)
            # part_index.append(res_k[1])
            # part_value.append(res_k[0])
            # self.index = tf.concat(part_index, -1)
            # self.value = tf.concat(part_value, -1)

            # self.logits = tf.nn.bias_add(tf.concat(part_res, -1), nce_biases)



            # [batch_size * seq_length, music_num]
            # labels_one_hot = tf.one_hot(neg_labels, self.music_num)
            # labels_one_hot = tf.reshape(labels_one_hot, [-1, self.music_num])
            # # [batch_size * seq_length]
            # loss = tf.nn.sigmoid_cross_entropy_with_logits(
            #     labels=labels_one_hot,
            #     logits=logits)
            # loss = tf.reshape(loss, [-1, self.sequence_length])
            # self.loss_test = tf.reduce_mean(tf.boolean_mask(loss, mask))
            self.loss_test = tf.constant(0, dtype=tf.float32, shape=[])
            # [batch_size * seq_length]
            neg_labels = tf.reshape(self.music_id, [-1])
            # [batch_size * seq_length]
            hit = tf.nn.in_top_k(self.logits, neg_labels, self.top_k)
            hit = tf.reshape(hit, [-1, self.sequence_length])
            mask_hit = tf.boolean_mask(hit, mask)
            self.hit_shape = tf.shape(mask_hit)
            self.recall = tf.reduce_mean(tf.to_float(mask_hit))

            # [batch_size*seq_length, top_k]
            top_k_index = tf.nn.top_k(self.logits, self.top_k)[1]
            index_mask = tf.boolean_mask(top_k_index, tf.reshape(mask, [-1]))
            label_mask = tf.boolean_mask(neg_labels, tf.reshape(mask, [-1]))
            label_mask = tf.reshape(label_mask, [-1, 1])
            self.rank = tf.where(tf.equal(tf.to_int32(index_mask), tf.to_int32(label_mask)))[:, -1]
            # self.rank_shape = tf.shape(rank)
            # self.mrr = tf.reduce_mean(1/(rank + 1))



            



    def trainstep_op(self):
        with tf.variable_scope("train_step"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            optim = tf.train.AdamOptimizer(learning_rate=self.lr)
            grads_and_vars = optim.compute_gradients(self.loss)
            grads_and_vars_clip = [[tf.clip_by_value(g, -5, 5), v] for g, v in grads_and_vars]
            self.train_op = optim.apply_gradients(grads_and_vars_clip, global_step=self.global_step)

    def train(self, train, dev):
        self.init_op = tf.global_variables_initializer()
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            sess.run(self.init_op)
            for epoch in range(self.epoch_num):
                self.run_one_epoch(sess, train, dev, epoch, saver)




    def run_one_epoch(self, sess, train, dev, epoch, saver):
        num_batches = (len(train) + self.batch_size - 1) // self.batch_size
        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        batches = batch_yield(train, self.batch_size, self.user_musics_dic)
        for step, (data, pre) in enumerate(batches):
            sys.stdout.write(' processing: {} batch / {} batches.'.format(step + 1, num_batches)
                             + '\r')
            step_num = epoch * num_batches + step + 1
            feed_dict = self.get_feed_dict(data, pre, self.dropout_train, 1)
            _, loss_train = sess.run([self.train_op, self.loss], feed_dict=feed_dict)

            # precursor = sess.run(self.precursor, feed_dict=feed_dict)
            # for p in precursor:
            #     for pp in p:
            #         if pp > 1075363:
            #             print(pp)

            if step + 1 == 1 or (step + 1) % 300 == 0 or step + 1 == num_batches:
                self.logger.info(
                    '{} epoch {}, step {}, loss: {:.4}, global_step: {}'.format(start_time,
                                                                                epoch + 1,
                                                                                step + 1,
                                                                                loss_train,
                                                                                step_num))
            if step + 1 == num_batches:
                saver.save(sess, self.model_path, global_step=step_num)

        self.logger.info('===========validation / test===========')
        losses, hit_rates, mrrs = self.dev_one_epoch(sess, dev)
        loss = np.mean(losses)
        hit_rate = np.mean(hit_rates)
        mrr = np.mean(mrrs)
        LOSS.append(loss)
        HIT_RATE.append(hit_rate)
        MRR.append(mrr)
        self.logger.info("loss:" + str(LOSS))
        self.logger.info("hit rate:" + str(HIT_RATE))
        self.logger.info("mrr:" + str(MRR))






    def dev_one_epoch(self, sess, dev):
        """
        :param sess:
        :param dev:
        :return:
        """
        losses, hit_rates, mrrs = [], [], []
        num_batches = (len(dev) + self.batch_size_test - 1) // self.batch_size_test
        for step, (data, pre) in enumerate(batch_yield(dev, self.batch_size_test,
                                                       self.user_musics_dic)):
            sys.stdout.write(' processing: {} batch / {} batches.'.format(step + 1, num_batches)
                             + '\r')
            loss, hit_rate, mrr = self.predict_one_batch(sess, data, pre)
            losses.append(loss)
            hit_rates.append(hit_rate)
            mrrs.append(mrr)
        return losses, hit_rates, mrrs

    def predict_one_batch(self, sess, data, pre):
        feed_dict = self.get_feed_dict(data, pre, 1.0, 0)
        loss_test, hit_rate, mrr, hit_shape, rank_shape = sess.run([self.loss_test, self.recall, self.mrr, self.hit_shape, self.rank_shape],
                                           feed_dict=feed_dict)
        print(hit_shape, rank_shape)

        return loss_test, hit_rate, mrr

    def get_feed_dict(self, data, pre, dropout, train):
        """
        self.music_id = tf.placeholder(tf.int32, shape=[None, self.sequence_length],
                                       name="music_id")
        self.lengths = tf.placeholder(tf.int32, shape=[None],
                                         name="lengths")
        self.precursor = tf.placeholder(tf.int32, shape=[None, None], name="precursor")
        self.pre_length = tf.placeholder(tf.int32, shape=[None], name="pre_length")
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.is_train = tf.placeholder(dtype=tf.int32, shape=[], name="is_train")
        """
        word_ids, seq_len_list = pad_sequences(data, maxl=self.sequence_length,
                                               pad_mark=self.music_num-1)
        pre_ids, pre_len_list = pad_sequences(pre, pad_mark=self.music_num-1)
        # for i in pre_ids:
        #     for j in i:
        #         if j >= 1075364:
        #             print(j)

        feed_dict = {self.music_id: word_ids,
                     self.lengths: seq_len_list,
                     self.precursor: pre_ids,
                     self.pre_length: pre_len_list,
                     self.dropout: dropout,
                     self.is_train: train}
        return feed_dict


    def reshape_to_matrix(self, input_tensor):
        """Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix)."""
        ndims = input_tensor.shape.ndims
        if ndims < 2:
            raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
                             (input_tensor.shape))
        if ndims == 2:
            return input_tensor

        width = input_tensor.shape[-1]
        # [batch_size * sequence_length, hidden_dim]
        output_tensor = tf.reshape(input_tensor, [-1, width])
        return output_tensor


    def layer_norm(self, input_tensor, name=None):
        """Run layer normalization on the last dimension of the tensor."""
        return tf.contrib.layers.layer_norm(
            inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)


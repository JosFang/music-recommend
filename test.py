# import tensorflow as tf
# m = tf.constant([[0, 2, 1],[2, 0, 1]])  # matrix
# y = tf.constant([3,2])  # values whose indices should be found
# y = tf.reshape(y, [-1, 1])  # [[1], [2]]
# cols = 1/(1+tf.where(tf.equal(m, y))[:, -1])  # [2,0]
#
# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     init.run()
#     print(sess.run(cols))
#
# a = [1]
# a.extend([])
# print(a)

# a = [0, 1]
# import numpy as np
# print(np.mean(1/(np.array(a)+1)))
a = (1/3+1/8+1/6+1/2+1/7)/5
print(a)
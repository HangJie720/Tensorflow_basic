import tensorflow as tf

# 创建一个变量，初始化标量 0.
state = tf.Variable(0,name="counter")

# 创建一个op,其作用是使state 增加1

one = tf.constant(1)
new_value = tf.add(state,one)
update = tf.assign(state,new_value)

# 启动图后，变量必须先经过'初始化' (init)op初始化
init_op = tf.initialize_all_variables()

# 启动图，运行op
with tf.Session() as sess:
    #运行'init' op
    sess.run(init_op)
    #打印'state'的初始值
    print(sess.run(state))
    # 运行op,更新'state',并打印'state'
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))

import tensorflow as tf

# 进入一个交互式Tensorflow会话
sess = tf.InteractiveSession()

x = tf.Variable([1.0,2.0])
a = tf.constant([3.0,3.0])

# 使用初始化器 initializer op 的run()方法初始化'x'
x.initializer.run()

# 增加一个减法sub op,从'x'减去'a'.运行减法op,输出结果

sub = tf.sub(x,a)
print(sub.eval())

import tensorflow as tf

# 创建一个常量op,产生一个1x2矩阵，这个op被作为一个节点加到默认图中

# 构造器的返回值代表该常量op的返回值

matrix1 = tf.constant([[3.,3.]])

# 创建另外一个常量op,产生一个2x1矩阵

matrix2 = tf.constant([[2.],[2.]])

# 创建一个矩阵乘法 matmul op,把'matrix1'和'matrix2'作为输入
# 返回值'product' 代表矩阵乘法的结果

product = tf.matmul(matrix1,matrix2)

# 默认图现在有三个节点, 两个constant() op, 和一个matmul() op. 为了真正进行矩阵相乘运算, 并得到矩阵
#　乘法的 结果, 你必须在会话里启动这个图.

# 默认启动图
sess = tf.Session()

# 调用 sess 的 'run()' 方法来执行矩阵乘法 op, 传入 'product' 作为该方法的参数.
# 上面提到, 'product' 代表了矩阵乘法 op 的输出, 传入它是向方法表明, 我们希望取回

result = sess.run(product)
print(result)

# 任务完成，关闭会话
sess.close()
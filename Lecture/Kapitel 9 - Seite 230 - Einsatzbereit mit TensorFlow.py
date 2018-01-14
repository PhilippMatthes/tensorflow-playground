import tensorflow as tf

x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")
f = x * x * y + y + 2

with tf.Session() as sess:
    x.initializer.run()
    y.initializer.run()
    result1 = f.eval()

init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    result2 = f.eval()

print(result1, result2)

x1 = tf.Variable(1)
print(x1.graph is tf.get_default_graph())

graph = tf.Graph()
with graph.as_default():
    x2 = tf.Variable(2)

print(x2.graph is tf.get_default_graph())
print(x2.graph is graph)

w = tf.constant(3)
x = w + 2
y = x + 5
z = x * 3

with tf.Session() as sess:
    print(y.eval())
    print(z.eval())


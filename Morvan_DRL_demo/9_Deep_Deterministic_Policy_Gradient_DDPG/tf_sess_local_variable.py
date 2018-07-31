import tensorflow as tf


class Funcs:

    def __init__(self):
        self.a = tf.random_normal((1,))
        self.b = tf.random_normal((1,))
        self.sum = self.a + self.b
        self.sess = tf.Session()

    def show(self):
        a = self.sess.run(self.a)

        return a

    def learn(self):
        a = [0.5]
        b = [0.4]
        sum = self.sess.run(self.sum, {self.a: a, self.b: b})

        return sum


if __name__ == '__main__':
    fun = Funcs()
    print(fun.show())
    print(fun.learn())
    print(fun.show())
    print(fun.show())
    print(fun.show())

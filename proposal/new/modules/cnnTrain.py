### Start to traini and evaluate the model

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())

x_input = df_new['train']['x']
x_input = [np.array([
            np.float32(x_input.iloc[i].values)
        ])
    for i in range(x_input.shape[0])]
y_input = df_new['train']['y']
y_input = [np.array([
            np.float32(y_input.iloc[i].values)
        ])
    for i in range(y_input.shape[0])]
# y_input = [np.array([y_input.iloc[i].values]) for i in range(y_input.shape[0])]

# not use random input

for i in range(df['train'].shape[0]):
    if 0 == i % 100:
        train_accuracy = []
        for j in range(50):
            train_accuracy.append(accuracy.eval(feed_dict={
                    keep_prob: 1,
                    x: x_input[i+j],#x_input.iloc[i+j].values, #
                    y_: y_input[i+j]#y_input.iloc[i+j].values #
                })
            )
        print "step {}, training accuracy {}".format(i, np.mean(train_accuracy))
    train_step.run(feed_dict={
        keep_prob: 0.5,
        x:  x_input[i],#x_input.iloc[i].values, #
        y_: y_input[i]#y_input.iloc[i].values#
    })

print "finish train for CNN"



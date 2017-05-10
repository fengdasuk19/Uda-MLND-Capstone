x_input = df_new['test']['x']
x_input = [np.array([
            np.float32(x_input.iloc[i].values)
        ])
    for i in range(x_input.shape[0])]
y_input = df_new['test']['y']
y_input = [np.array([
            np.float32(y_input.iloc[i].values)
        ])
    for i in range(y_input.shape[0])]

for i in range(df['test'].shape[0]):
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

print "finish test for CNN"


# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Combine training and test sets
x = np.concatenate((x_train, x_test))
y = np.concatenate((y_train, y_test))

# Split into training and test sets with 80:20 ratio
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Ít sample thui chứ cháy máy
Xtrain = x_train[:200]
ytrain = y_train[:200]
Xtest = x_test[:100]
ytest = y_test[:100]
Xtrain = Xtrain/255
Xtest = Xtest/255
ytrain = LabelBinarizer().fit_transform(ytrain) 
ytest = LabelBinarizer().fit_transform(ytest)

#Train
l1 = Conv2D(kernel_size=3, filter=32, pad=1)
l2 = MaxPooling(2, 2)
""" l3 = Conv2D(kernel_size=3, filter=64, pad=1)
l4 = MaxPooling(2, 2) """
l5 = Flatten()
l6 = Dense(512, active="relu")
l7 = Dense(ytrain.shape[1], active="softmax")
md = Sequential(l1, l2, l5, l6, l7)
sgd = SGD(batch_size=20, max_epoch=5, eta=0.1, momentum=0.9)
md.fit(Xtrain, ytrain, optimizer=sgd)

# Score of test set
print(md.emualate(Xtest, ytest))

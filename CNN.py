from layers.Conv2D import Conv2D
from layers.MaxPooling import MaxPooling
from layers.Flatten import Flatten
from layers.Dense import Dense
from optimizers.Adam import Adam
from optimizers.SGD import SGD
from optimizers.RMSProp import RMSProp
from sklearn.preprocessing import LabelBinarizer
from Sequential import Sequential
import os, cv2, numpy as np, matplotlib.pyplot as plt

np.set_printoptions(4, suppress=True)

data = []
X_train, y_train =  [], []
folder = ['4', '7', '11', '15']
people = ["Ribi", "Suboi", "Phạm Huy Hoàng", "Mai Phương Thúy"] #Name of the popular Vietnamese people

for f in folder:
    path = "./dataset/VN-Celeb/" + f
    for filename in os.listdir(path):
        file_path = path + '/' + filename
        img = cv2.imread(file_path, 0)
        if img is None:
            break
        img = cv2.resize(img, dsize=(128, 128))
        data.append((img, int(f)))

np.random.shuffle(data)
for d in data:
    X_train.append(d[0])
    y_train.append(d[1])

X_train = np.array(X_train).reshape((len(X_train), 1, 128, 128))
X_train = X_train/255
y_train = LabelBinarizer().fit_transform(y_train)

md = Sequential()
conv = Conv2D((3, 3), 4, 1, 'same', "relu")
md.add(conv)
md.add(MaxPooling(3, 3))
md.add(Flatten())
md.add(Dense(256, 'relu'))
md.add(Dense(y_train.shape[1], 'softmax')) 
md.compile(optimizer=Adam(0.001, beta1=0.9, beta2=0.999))
md.fit(X_train[:143], y_train[:143], batch_size=10, epochs=4)
_, score = md.evalute(X_train[143:], y_train[143:])
print(f"Test accuracy {score * 100:.2f}%")
print(conv.get_weight())

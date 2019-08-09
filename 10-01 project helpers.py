from tensorflow.keras.datasets import fashion_mnist
from scipy.misc import imsave

(X_train,y_train), (X_test, y_test) = fashion_mnist.load_data()

for i in range(5):
    imsave(name=f"uploads/{i}.png", arr = X_test[i])
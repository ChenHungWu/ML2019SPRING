import numpy as np
import itertools
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.metrics import confusion_matrix

model_path = 'en05.h5'
X_train_valid = 'X_train_valid.npy'
Y_train_valid_label = 'Y_train_valid_label.npy'

classes = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"] #list of class

def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.jet):
    """
    This function prints and plots the confusion matrix.
    """
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix.png')
    plt.show()

model = load_model(model_path)
X_train_valid, Y_train_valid_label = np.load(X_train_valid), np.load(Y_train_valid_label)
np.set_printoptions(precision=2)

predictions = model.predict_classes(X_train_valid)
conf_mat = confusion_matrix(y_true=Y_train_valid_label, y_pred=predictions)
plt.figure()
plot_confusion_matrix(conf_mat, classes=classes)
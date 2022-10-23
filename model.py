import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

class recog_majan():
    def __init__(self,model_name, file_name, threshold=0.85):
        self.image_size = 64
        self.batch_size = 1
        self.model = self._load_model(model_name, file_name) 
        self.labels = self._load_labels('./label/labels2.txt')
        self.threshold = threshold

    def _load_model(self,model_name, file_name="./"):
        return   tf.keras.models.load_model(file_name+"/"+model_name)

    def _load_labels(self,file_path='./label/labels.txt'):
        labels = []
        with open(file_path,'r',encoding="utf-8") as f:
            for line in f:
                labels.append(line.rstrip())
        print(labels)
        return labels

    def predict_dir(self,file_path):
        test_data = self._load_data(file_path)
        pred = self.model.predict(test_data)
        print(self.labels[pred.argmax()])
        return pred

    def predict(self,file_path):
    #tensorflowの場合、テンソルに変換して正規化やサイズ変更、値の変更が必要になる
        # img = tf.io.read_file(file_path)
        # img = tf.convert_to_tensor(file_path)
        # img = np.asarray(file_path).astype("float32")
        # img = torchvision.transforms.functional.to_tensor(file_path)
        # # image = tf.image.decode_image(img, channels=3)
        # image = tf.image.resize(img, [64,64])


        #streamlit の場合
        img_array = np.array(file_path)
        image = tf.image.resize(img_array, size=(64,64))
        image /= 255.0
        image = np.expand_dims(image, axis=0)
        pred = self.model.predict(image)
        return self.labels[pred.argmax()]

    def _load_data(self, file_path):
        data_gen = ImageDataGenerator(rescale=1./255)
        data = data_gen.flow_from_directory(
        file_path, target_size=(self.image_size,self.image_size),
        color_mode="rgb", batch_size=self.batch_size, class_mode="categorical",
        shuffle=False
        )
        return data

    def calc_score(self,file_path):
        validation_data = self._load_data(file_path)

        #modelの評価
        score = self.model.evaluate(validation_data)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        if self.threshold > score[1]:
            print("score is low")

        return score

class train_model:
    def __init__(self, epochs=12 ,batch_size=8,learning_rate=0.001,vis=1):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.train_data, self.validation_data = self.load_data()
        self.vis = 1

    def load_data(self):

      train_data_gen = ImageDataGenerator(rescale=1./255)
      val_data_gen = ImageDataGenerator(rescale=1./255)

      train_data = train_data_gen.flow_from_directory(
      train_dir, target_size=(IMAGE_SIZE, IMAGE_SIZE),
      color_mode='rgb', batch_size=self.batch_size,
      class_mode='categorical', shuffle=True)

      validation_data = val_data_gen.flow_from_directory(
      val_dir, target_size=(IMAGE_SIZE, IMAGE_SIZE),
      color_mode='rgb', batch_size=self.batch_size,
      class_mode='categorical', shuffle=False)

      return train_data, validation_data

    def set_model(self):
        model = Sequential() #線形モデル
        model.add(Conv2D(12, (6, 6), padding='same',
            input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))) ##畳み込み
        model.add(MaxPooling2D(pool_size=(4, 4))) #プーリング層
        model.add(Activation('relu')) #活性化関数
        model.add(Conv2D(12, (6, 6))) #畳み込み層
        model.add(Activation('relu')) ##活性化関数
        model.add(MaxPooling2D(pool_size=(4, 4))) #プーリング層
        model.add(Dropout(0.15))
    
        model.add(Flatten()) #平滑化　データの一次元化
        model.add(Dense(128)) #層
        model.add(Activation('relu')) #活性化関数
        model.add(Dense(128)) #層
        model.add(Activation('relu')) #活性化関数
        model.add(Dense(64)) #層
        model.add(Activation('relu')) #活性化関数
        model.add(Dropout(0.15))
        model.add(Dense(NUM_CLASSES)) #層
        model.add(Activation('softmax'))
    
        self.model = model

    def complie(self):
        opt = tf.keras.optimizers.Adam(lr=self.learning_rate) #最適化
    #opt = tf.keras.optimizers.SGD(lr=LEARNING_RATE)

        self.model.compile(opt, loss='categorical_crossentropy',
        metrics=['accuracy'])
        

    def fit(self,cb=[]):
        self.history= self.model.fit(self.train_data, epochs=self.epochs, validation_data= self.validation_data, verbose=1, callbacks=cb)

    def eval(self):
        score = self.model.evaluate(self.validation_data)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        return [score[0],score[1],self.epochs, self.batch_size, self.learning_rate,model]

    def save_model(self,model_name="my_model.h5"):
         # Save model
      save_model_path = os.path.join("./model/", model_name)
      self.model.save(save_model_path)

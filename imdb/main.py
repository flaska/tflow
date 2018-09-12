import data
import model
import plot
from tensorflow import keras
import tensorflow as tf

(train_data, train_labels), (test_data, test_labels) = data.get_data()
reverse_word_index = data.get_reverse_word_index()

(reviews_train, labels_train), (reviews_val, labels_val) = data.getXyValAndTrain()


# creating and saving model

#model = model.get_model()

#history = model.fit(reviews_train,
#                     labels_train,
#                     epochs=40,
#                     batch_size=512,
#                     validation_data=(reviews_val, labels_val),
#                     verbose=1)
#
# plot.display_acc(history)
#
# model.save('my_model.h5')

# loading saved model
model = keras.models.load_model('my_model.h5')
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()
# load end

results = model.evaluate(test_data, test_labels)

print(results)

predictions = model.predict(test_data[0:5])
print(predictions)

for i in range(0, 5):
    print(data.decode_review(test_data[i]))
    print(predictions[i])
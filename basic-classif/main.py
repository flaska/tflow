import data
import model
import display

(train_images, train_labels), (test_images, test_labels) = data.get_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# display.display_sample(train_images, class_names, train_labels)

model = model.get_model()
model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

predictions = model.predict(test_images)

display.plot_results(predictions, test_images, test_labels)
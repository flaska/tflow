from tensorflow import keras

imdb = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

word_index = imdb.get_word_index()
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)


def get_data():
    return (train_data, train_labels), (test_data, test_labels)

def get_word_index():
    return word_index

def get_reverse_word_index():
    return reverse_word_index

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


reviews_val = train_data[:10000]
reviews_train = train_data[10000:]

labels_val = train_labels[:10000]
labels_train = train_labels[10000:]

def getXyValAndTrain():
    return (reviews_train, labels_train), (reviews_val, labels_val)
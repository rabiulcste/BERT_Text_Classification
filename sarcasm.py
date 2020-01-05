# We will use the official tokenization script created by the Google team

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow_hub as hub
from sklearn import metrics

import tokenization
import model_configs

def bert_encode(texts, tokenizer, max_len=512):
    all_tokens = []
    all_masks = []
    all_segments = []

    for text in texts:
        text = tokenizer.tokenize(text)
        text = text[:max_len - 2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)

        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        tokens += [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len

        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)

    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)


def build_model(bert_layer, max_len=512):
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

    _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    clf_output = sequence_output[:, 0, :]
    out = Dense(1, activation='sigmoid')(clf_output)

    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    model.compile(Adam(lr=2e-5), loss='binary_crossentropy', metrics=['accuracy'])

    return model


module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1"
bert_layer = hub.KerasLayer(module_url, trainable=True)

train = pd.read_csv(model_configs.FRACK_TRAIN)
test = pd.read_csv(model_configs.FRACK_TEST)
# submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

train_input = bert_encode(train.clean_text.values, tokenizer, max_len=50)
test_input = bert_encode(test.clean_text.values, tokenizer, max_len=50)
train_labels = train.label.values
test_labels = test.label.values

model = build_model(bert_layer, max_len=50)
model.summary()

checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)

train_history = model.fit(
    train_input, train_labels,
    validation_split=0.2,
    epochs=3,
    callbacks=[checkpoint],
    batch_size=64
)

model.load_weights('model.h5')

score, acc = model.evaluate(test_input, test_labels, batch_size=128)
print('Test score:', score)
print('Test accuracy:', acc)

#
# test_pred = model.predict(test_input, batch_size=32, verbose=2)
# y_pred = test_pred.round().astype(int)
#
# print('Accuracy::', metrics.accuracy_score(y, y_pred))
# print('Precision::', metrics.precision_score(y, y_pred, average='weighted'))
# print('Recall::', metrics.recall_score(y, y_pred, average='weighted'))
# print('F_score::', metrics.f1_score(y, y_pred, average='weighted'))
# print('F_score::', metrics.classification_report(y, y_pred))

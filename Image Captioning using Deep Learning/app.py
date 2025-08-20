import os
import sys
import pickle
import numpy as np
import tensorflow as tf
from PIL import Image
import gradio as gr

from tensorflow.keras import preprocessing
# sys.modules['keras'] = keras
sys.modules['keras.src.preprocessing'] = preprocessing

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.densenet import DenseNet201, preprocess_input
from tensorflow.keras.layers import Layer, Dense, Dropout, LayerNormalization, MultiHeadAttention

try:
    from tensorflow.keras.saving import register_keras_serializable
except ImportError:
    from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable()
class TransformerEncoder(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation='relu'),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

vocab_size = len(tokenizer.word_index) + 1
max_length = 34  # Change if needed


model = tf.keras.models.load_model(
    "caption_model.keras",
    custom_objects={"TransformerEncoder": TransformerEncoder}
)


base_model = DenseNet201(weights='imagenet', include_top=False, pooling='avg')
model_cnn = tf.keras.Model(inputs=base_model.input, outputs=base_model.output)


def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def beam_search_decoder(model, image, tokenizer, max_length, beam_width=3):
    start = 'startseq'
    sequences = [[start, 0.0]]
    for _ in range(max_length):
        all_candidates = []
        for seq, score in sequences:
            sequence = tokenizer.texts_to_sequences([seq])[0]
            sequence = pad_sequences([sequence], maxlen=max_length, padding='post')
            yhat = model.predict([image, sequence], verbose=0)[0]
            top_k = np.argsort(yhat)[-beam_width:]
            for word_id in top_k:
                word = idx_to_word(word_id, tokenizer)
                if word is None:
                    continue
                candidate = seq + ' ' + word
                candidate_score = score - np.log(yhat[word_id] + 1e-10)
                all_candidates.append([candidate, candidate_score])
        sequences = sorted(all_candidates, key=lambda tup: tup[1])[:beam_width]
    for seq, score in sequences:
        if 'endseq' in seq:
            return seq
    return sequences[0][0]


def predict_caption(image):
    image = image.resize((224, 224))
    img = img_to_array(image)
    img = img.reshape((1, 224, 224, 3))
    img = preprocess_input(img)
    feature = model_cnn.predict(img, verbose=0)
    caption = beam_search_decoder(model, feature, tokenizer, max_length)
    return caption.replace("startseq", "").replace("endseq", "").strip()


def caption_with_image(image):
    caption = predict_caption(image)
    return [image, caption]

gr.Interface(
    fn=caption_with_image,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Image(type="pil", label="Uploaded Image"), gr.Textbox(label="Generated Caption")],
    title="Image Captioning Demo",
    description="Upload an image to generate a caption using DenseNet201 + Transformer + LSTM."
).launch(debug=True)

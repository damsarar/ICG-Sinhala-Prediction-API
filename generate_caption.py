from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from pickle import load
import numpy as np

model_dir = "saved_model/MSCOCO_model_v1_29"
model = load_model(model_dir)

# loading the encoded image features from the pickle file
index_to_word = load(open("index_to_word.p", "rb"))
word_to_index = load(open("word_to_index.p", "rb"))


# function to generate a caption for a given image
def generate_caption(test_image, maximum_caption_length):
    # asigning the SOS tag as the first input word
    input_word = 'SOS'

    # loop through maximum caption length times
    for i in range(maximum_caption_length):
        # extracting index of words in the sequence using word to index dictionary
        sequence = [word_to_index[w]
                    for w in input_word.split() if w in word_to_index]

        # print(sequence)

        # pading the sequence until the maximum caption length
        sequence = pad_sequences([sequence], maxlen=maximum_caption_length)

        # predicting the next word for the given image and the sequence
        y_pred = model.predict([test_image, sequence], verbose=0)

        # getting the index of the maximum probability
        y_pred_index = np.argmax(y_pred)

        # extracting the word for the predicted index using index to word dictionary
        word = index_to_word[y_pred_index]

        # appending the new word to the input word
        input_word += ' ' + word

        # checking for the EOS tag to stop creating the sequence
        if word == 'EOS':
            break

    # creating the final sequence after removing the SOS, EOS tags
    final_generated_caption = input_word.split()[1:-1]
    final_generated_caption = ' '.join(final_generated_caption)

    return final_generated_caption

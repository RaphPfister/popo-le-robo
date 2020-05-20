'''
Main script to launch bot
Requires a functional Telegram bot and its API token stored as env variable API_TOKEN (Shout @BotFather on telegram)
Requires a trained tensorflow.keras.preprocessing.text.Tokenizer stored as "./tokenizer.pickle"
Requires a trained tensorflow.keras.models.Sequential stored as "./model.hdf5"
'''

import os
import pickle

import numpy as np

import telebot
from tensorflow.keras.models import load_model

API_TOKEN = os.environ["API_TOKEN"]

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
model = load_model('model.hdf5')

# Instantiating bot
bot = telebot.TeleBot(API_TOKEN, threaded=False)


def generate(seed, max_length, n):
    """
    Generates a full sentence fiven a starting text 'seed'
    e.g. generate("Bonjour je m'appelle", 3, 3) may return "Bonjour, je m'appelle Raphaël et toi"
    Iteratively appends predicted word to said string and remake the prediction,
    Stops when end of sentence <eos> token is predicted as next word
    :param seed: [String] Starting seed to give the model
    :param max_length: [int] max number of words to predict
    :param diversity: [int] diversity parameter, next word will randomly chosen between the 'n'-best probable next words
    :return: [string] generated seed
    """
    print("SEED: ",seed)
    n = min(n, 150) # Avoids crash at high diversity value
    seed_tk = tokenizer.texts_to_sequences([seed])
    for i in range(max_length):
        y_pred = model.predict(seed_tk)
        next_words_proba = y_pred[0][-1]
        best_n_next = next_words_proba.argsort()[-n:]
        next_word = np.random.choice(best_n_next)
        y_text = tokenizer.index_word[next_word]
        if y_text == '<eos>':
            return seed
        seed = seed + ' ' + y_text
        seed_tk = tokenizer.texts_to_sequences([seed])
    return seed


@bot.message_handler(commands=['imite'])
def imite(message):
    """
    Syntax : /imite [int: diversity] seed
    Sends generated sentence to chat, given diversity (=n). Diversity is optionnal and will default to 3.
    :param message: User message
    :return: None
    """

    print(message.text)
    splitted = message.text.split(' ')
    if len(splitted)>2 and splitted[1].isnumeric():
        diversity_num = int(message.text.split(' ')[1])
        bot.reply_to(message, generate(message.text.replace('/imite {} '.format(message.text.split(' ')[1]), ''), 50, n=diversity_num))
    else:
        div = np.random.randint(1,7)
        bot.reply_to(message, generate(message.text.replace('/imite ', ''), 50, n=3))

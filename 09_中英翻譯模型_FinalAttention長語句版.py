import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Attention, Concatenate
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 定義訓練語料庫：問題與答案對應關係
questions = ["高雄 花蓮 苗栗 台中 嘟嘟好  新北市 台北在哪裡 台灣北部 新竹 金門 澎湖"]
answers = ["台北"]

# 為答案加上開始 (\t) 與結束 (\n) 標記，幫助模型識別起訖
target_texts = ['\t' + ans + '\n' for ans in answers]

# 初始化輸入端(問題)分詞器，使用字元級分詞
input_tokenizer = Tokenizer(char_level = True)
input_tokenizer.fit_on_texts(questions)
encoder_input_data = input_tokenizer.texts_to_sequences(questions)

# 初始化輸出端(答案)分詞器
target_tokenizer = Tokenizer(char_level = True)
target_tokenizer.fit_on_texts(target_texts)
decoder_input_data = target_tokenizer.texts_to_sequences(target_texts)

# 計算最大長度：取得序列的最長長度作為填充標準
max_encoder_seq_length = max([len(txt) for txt in encoder_input_data])
max_decoder_seq_length = max([len(txt) for txt in decoder_input_data])

# 計算總字元數
num_encoder_tokens = len(input_tokenizer.word_index) + 1
num_decoder_tokens = len(target_tokenizer.word_index) + 1

# 規劃相同長度：使用 post padding 補齊序列
encoder_input_data = pad_sequences(encoder_input_data, maxlen = max_encoder_seq_length, padding = 'post')
decoder_input_data = pad_sequences(decoder_input_data, maxlen = max_decoder_seq_length, padding = 'post')

# 建立訓練用的目標數據：將序列往後偏移一格作為預測目標 (One-hot 格式)
decoder_target_data = np.zeros((len(questions), max_decoder_seq_length, num_decoder_tokens), dtype = 'float32')
for i, seq in enumerate(decoder_input_data):
    for t, char_id in enumerate(seq):
        if t > 0:
            decoder_target_data[i, t - 1, char_id] = 1.0

# 增加維度以避免過度壓縮與資訊孤島問題，提升特徵表達能力
latent_dim = 12

# 規劃 encoder 編碼處理
encoder_inputs = Input(shape = (None,))
en_emb = Embedding(num_encoder_tokens, 32)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_sequences = True, return_state = True, name = 'encoder_lstm')
encoder_outputs, state_h, state_c = encoder_lstm(en_emb)
encoder_states = [state_h, state_c]

# 規劃 decoder 解碼處理
decoder_inputs = Input(shape = (None,))

# 先定義層物件，確保訓練與推論使用相同的 Embedding 與 LSTM 權重
de_emb = Embedding(num_decoder_tokens, 32)(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences = True, return_state = True)
decoder_outputs, _, _ = decoder_lstm(de_emb, initial_state = encoder_states)

# 增加注意力機制，讓模型能夠專注於輸入序列的相關部分，提高翻譯品質
attention_layer = Attention(name = 'attention_layer')

# 計算注意力權重，將 decoder 的輸出與 encoder 的輸出進行對齊，獲取上下文向量
context_vector = attention_layer([decoder_outputs, encoder_outputs])

# 將上下文向量與 decoder 的輸出結合，提供更多資訊給 Dense 層進行預測
decoder_combined_context = Concatenate(axis = -1)([decoder_outputs, context_vector])

# 定義 Dense 層物件，確保訓練與推論使用相同的權重
decoder_dense = Dense(num_decoder_tokens, activation = 'softmax')
decoder_outputs = decoder_dense(decoder_combined_context)

# 建立訓練模型
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy')
model.summary()
model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size = 1, epochs = 100, verbose = 0)

print("訓練完成！")

# 建立推論用的編碼器模型
encoder_model = Model(encoder_inputs, [encoder_outputs, state_h, state_c])

# 規劃 decoder 解碼處理，輸入為前一個時間步的輸出與編碼器的狀態
decoder_inputs = Input(shape = (1,))

# 從訓練好的主模型中擷取層來重用權重
decoder_emb = Embedding(num_decoder_tokens, 32)(decoder_inputs)
state_input_h = Input(shape = (latent_dim,))
state_input_c = Input(shape = (latent_dim,))
encoder_outputs_input = Input(shape = (max_encoder_seq_length, latent_dim))
decoder_outputs, inf_h, inf_c = decoder_lstm(
    decoder_emb, initial_state=[state_input_h, state_input_c])

# 推論階段的注意力機制：將 decoder 的輸出與 encoder 的輸出進行對齊，獲取上下文向量
inf_context = attention_layer([decoder_outputs, encoder_outputs_input])
inf_combined = Concatenate(axis = -1)([decoder_outputs, inf_context])
inf_final = decoder_dense(inf_combined)

decoder_model = Model(
    [decoder_inputs, state_input_h, state_input_c, encoder_outputs_input],
    [inf_final, inf_h, inf_c]
)

# 規劃預測函數
def decode_sequence(input_seq):
    # 執行預測取得狀態向量
    encout, h, c = encoder_model.predict(input_seq)

    # 設定初始輸入標記為 \t (開始符號)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = target_tokenizer.word_index['\t']

    stop_condition = False
    decoded_sentence = ''

    while not stop_condition:
        print('原本推論的decoder模型需要兩個參數，現在要改為四個')
        print('encoder的輸出帶到decoder的輸入')

        output_tokens, h, c = decoder_model.predict([target_seq, h, c, encout])

        # 取得機率最大的下一個字元索引
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = target_tokenizer.index_word.get(sampled_token_index, '')

        # 若預測到 \n (結束) 或達到最大長度則停止
        if sampled_char == '\n' or len(decoded_sentence) > 20:
            stop_condition = True
        else:
            decoded_sentence += sampled_char

        # 更新輸入與狀態
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]

    return decoded_sentence

# 測試預估結果
test_input = input_tokenizer.texts_to_sequences(["首都是哪裡"])
test_input = pad_sequences(test_input, maxlen = max_encoder_seq_length, padding = 'post')

print(f"測試輸入: 首都是哪裡")
print(f"模型預測答案: {decode_sequence(test_input)}")
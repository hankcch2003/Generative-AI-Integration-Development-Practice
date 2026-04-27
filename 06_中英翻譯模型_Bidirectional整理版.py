import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Bidirectional, Concatenate
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import plot_model

# 定義訓練語料庫：問題與答案對應關係
questions = ["台灣首都是台北", "首都是哪裡", "101大樓"]
answers = ["台北", "台北", "台北"]

# 為答案加上開始 (\t) 與結束 (\n) 標記，幫助模型識別起訖
target_texts = ['\t' + ans + '\n' for ans in answers]

print('以下為字串序列化處理過程，計算出最大長度後規劃相同長度')

# 初始化輸入端(問題)分詞器，使用字元級分詞
input_tokenizer = Tokenizer(char_level = True) 
input_tokenizer.fit_on_texts(questions)
X = input_tokenizer.texts_to_sequences(questions)

# 初始化輸出端(答案)分詞器
target_tokenizer = Tokenizer(char_level = True)
target_tokenizer.fit_on_texts(target_texts)
y = target_tokenizer.texts_to_sequences(target_texts)

# 計算最大長度：取得英中序列的最長長度作為填充標準
max_len_eng = max(len(seq) for seq in X)
max_len_chi = max(len(seq) for seq in y)

# 規劃相同長度：使用 post padding 補齊序列，確保輸入維度一致
X = pad_sequences(X, maxlen = max_len_eng, padding = 'post')
y = pad_sequences(y, maxlen = max_len_chi, padding = 'post')

# 建立訓練用的目標數據：將序列往後偏移一格作為預測目標 (One-hot 格式)
num_decoder_tokens = len(target_tokenizer.word_index) + 1
decoder_target_data = np.zeros((len(questions), max_len_chi, num_decoder_tokens), dtype='float32')
for i, seq in enumerate(y):
    for t, char_id in enumerate(seq):
        if t > 0:
            decoder_target_data[i, t - 1, char_id] = 1.0

# 增加維度以避免過度壓縮與資訊孤島問題，提升特徵表達能力
latent_dim = 8 

# 規劃 encoder 編碼處理，也就是英文資料
encoder_inputs = Input(shape = (None,))

# 先定義層物件，方便後續推論時重用權重
en_emb_layer = Embedding(len(input_tokenizer.word_index) + 1, 32)
en_emb = en_emb_layer(encoder_inputs)

# 定義 Encoder：使用雙向 LSTM 強化前後文特徵提取
encoder_lstm = Bidirectional(LSTM(latent_dim, return_state = True))

# 修改 Layer 輸出的項目會增加，因為前後關係
encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder_lstm(en_emb)

# 長短期記憶就會有往前與往後兩種，state_h 與 state_c 得合併前後資訊
state_h = Concatenate()([forward_h, backward_h])
state_c = Concatenate()([forward_c, backward_c])
encoder_states = [state_h, state_c]

# 規劃 decoder 解碼處理，也就是中文
decoder_inputs = Input(shape = (None,))

# 先定義層物件，確保訓練與推論使用相同的 Embedding 與 LSTM 權重
de_emb_layer = Embedding(num_decoder_tokens, 32)
de_emb = de_emb_layer(decoder_inputs)

# 因為有前後的 Encoder 資訊，所以神經元數量要乘以 2
decoder_lstm = LSTM(latent_dim * 2, return_sequences = True, return_state = True)
de_outputs, _, _ = decoder_lstm(de_emb, initial_state = encoder_states)

# 此處不使用 Attention 運算，直接將 decoder_outputs 送入全連接層預測字元機率分布
decoder_dense = Dense(num_decoder_tokens, activation = 'softmax')
outputs = decoder_dense(de_outputs)

model = Model([encoder_inputs, decoder_inputs], outputs)
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy')

# 產生主模型架構圖
plot_model(model, to_file = 'model3.png', show_shapes = True, show_layer_names = True)

# 訓練模型：執行 100 次訓練 (Epochs)，使模型能充分學習並記憶小樣本數據的對應關係
model.fit([X, y], decoder_target_data, batch_size = 1, epochs = 100, verbose = 0)

print("訓練完成！")

# 建立推論用的模型
encoder_model = Model(encoder_inputs, encoder_states)

# 獨立的推論用 Decoder 模型結構
latent_dim2 = latent_dim * 2
decoder_state_input_h = Input(shape = (latent_dim2,))
decoder_state_input_c = Input(shape = (latent_dim2,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

# 從訓練好的主模型中擷取層來重用權重
inf_de_emb = de_emb_layer(decoder_inputs)
inf_de_outputs, inf_h, inf_c = decoder_lstm(inf_de_emb, initial_state = decoder_states_inputs)
inf_states = [inf_h, inf_c]
inf_outputs = decoder_dense(inf_de_outputs)

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [inf_outputs] + inf_states
)

# 規劃預估函數
def decode_sequence(input_text):
    # 將輸入文本轉換為序列格式
    seq = input_tokenizer.texts_to_sequences([input_text])
    seq = pad_sequences(seq, maxlen = max_len_eng, padding = 'post')
    
    # 執行預測取得狀態向量
    states_value = encoder_model.predict(seq, verbose = 0)

    # 設定初始輸入標記為 \t (開始符號)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = target_tokenizer.word_index['\t']
    
    stop_condition = False
    decoded_sentence = ''
    
    while not stop_condition:
        # 逐步進行解碼預測
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value, verbose = 0)
        
        # 取得機率最大的下一個字元索引
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = target_tokenizer.index_word.get(sampled_token_index, '')
        
        # 若預測到 \n (結束) 或達到最大長度則停止
        if sampled_char == '\n' or len(decoded_sentence) > 20:
            stop_condition = True
        else:
            decoded_sentence += sampled_char
            
        # 更新輸入與狀態，為下一個字元的預測做準備
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]
        
    return decoded_sentence

# 測試預估結果
test_q = "首都是哪裡"

print("-" * 30)
print(f"測試輸入: {test_q}")
print(f"模型預測答案: {decode_sequence(test_q)}")
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
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

# 增加維度以避免過度壓縮與資訊孤島問題，提升特徵表達能力
latent_dim = 64

print('規劃 encoder 編碼處理，也就是英文資料')

# 定義 Encoder：負責將輸入轉換為上下文特徵與狀態向量 (state_h, state_c)
encoder_inputs = Input(shape = (max_len_eng,))
en_emb_layer = Embedding(len(input_tokenizer.word_index) + 1, 64)
en_x = en_emb_layer(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state = True)
_, state_h, state_c = encoder_lstm(en_x)
encoder_states = [state_h, state_c]

print('規劃 decoder 解碼處理，也就是中文')

# 定義 Decoder：接收編碼器的狀態，並逐步產生對應的中文字
decoder_inputs = Input(shape = (None,))
de_emb_layer = Embedding(len(target_tokenizer.word_index) + 1, 64)
de_x = de_emb_layer(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences = True, return_state = True)
de_outputs, _, _ = decoder_lstm(de_x, initial_state = encoder_states)

print('假設不加入 Attention')

# 此處不使用 Attention 運算，直接將 decoder_outputs 送入全連接層預測字元機率分布
decoder_dense = Dense(len(target_tokenizer.word_index) + 1, activation = 'softmax')
outputs = decoder_dense(de_outputs)

model = Model([encoder_inputs, decoder_inputs], outputs)
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy')
model.summary()

# 產生主模型架構圖
plot_model(model, to_file = 'model_main.png', show_shapes = True, show_layer_names = True)

# 訓練模型：執行 300 次訓練 (Epochs)，使模型能充分學習並記憶小樣本數據的對應關係
# 準備目標數據：將 y 序列往後偏移一格作為預測目標
y_target = np.zeros_like(y)
y_target[:, :-1] = y[:, 1:]
y_target = np.expand_dims(y_target, -1)

model.fit(
    [X, y],
    y_target,
    epochs = 300, 
    verbose = 0
)

print('規劃評估模型，評估不等於訓練')

# 獨立的推論用 Encoder 模型
encoder_model = Model(encoder_inputs, encoder_states)

# 獨立的推論用 Decoder 模型結構
decoder_state_input_h = Input(shape = (latent_dim,))
decoder_state_input_c = Input(shape = (latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

# 從訓練好的主模型中擷取層來重用權重
inf_de_x = de_emb_layer(decoder_inputs)
inf_de_outputs, inf_h, inf_c = decoder_lstm(inf_de_x, initial_state = decoder_states_inputs)
inf_outputs = decoder_dense(inf_de_outputs)

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [inf_outputs, inf_h, inf_c]
)

# 產生推論模型的結構圖
plot_model(encoder_model, to_file = 'encoder_model.png', show_shapes = True, show_layer_names = True)
plot_model(decoder_model, to_file = 'decoder_model.png', show_shapes = True, show_layer_names = True)

print('規劃預估函數')
def decode_sequence(input_text):
    # 將輸入文本轉換為序列格式
    seq = input_tokenizer.texts_to_sequences([input_text])
    seq = pad_sequences(seq, maxlen = max_len_eng, padding = 'post')
    
    # 執行預測並隱藏進度條，使輸出結果更簡潔
    states = encoder_model.predict(seq, verbose = 0)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = target_tokenizer.word_index['\t'] # 設定初始標記為開始符號
    
    stop = False
    predicted_chars = []
    
    while not stop:
        # 逐步進行解碼預測
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states, verbose = 0
        )
        
        # 取得機率最大的下一個字元索引
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = target_tokenizer.index_word.get(sampled_token_index, "")
        
        # 若預測到結束符號或達到最大長度則停止
        if sampled_char == '\n' or len(predicted_chars) >= max_len_chi:
            stop = True
            break
            
        if sampled_char != "":
            predicted_chars.append(sampled_char)
        
        # 更新輸入與狀態，為下一個字元的預測做準備
        target_seq = np.array([[sampled_token_index]])
        states = [h, c]

    # 格式化輸出：直接結合預測出的字元
    return "".join(predicted_chars)

# 測試預估結果
test_input = "首都是哪裡"
result = decode_sequence(test_input)

print("-" * 30)
print("測試輸入:", test_input)
print("模型預測:", result)
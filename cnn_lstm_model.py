import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

class CNNLSTMModel:
    def __init__(self, config):
        self.config = config
        self.model = self.build_model()

    def build_model(self):
        model = Sequential([
            Conv1D(filters=self.config.CNN_FILTERS_1, kernel_size=self.config.CNN_KERNEL_SIZE, padding='same', activation='relu', input_shape=(self.config.WINDOW_SIZE, self.config.TOTAL_FEATURES)),
            BatchNormalization(),
            Conv1D(filters=self.config.CNN_FILTERS_2, kernel_size=self.config.CNN_KERNEL_SIZE, padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            LSTM(units=self.config.LSTM_UNITS_1, return_sequences=True, dropout=self.config.DROPOUT_RATE),
            BatchNormalization(),
            LSTM(units=self.config.LSTM_UNITS_2, return_sequences=False, dropout=self.config.DROPOUT_RATE),
            BatchNormalization(),
            Dense(units=self.config.DENSE_UNITS, activation='relu'),
            Dropout(self.config.DROPOUT_RATE),
            Dense(units=1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(learning_rate=self.config.LEARNING_RATE), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train(self, train_data, val_data):
        X_train, y_train = train_data
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=self.config.EARLY_STOPPING_PATIENCE, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=self.config.REDUCE_LR_PATIENCE)
        ]
        print(f"   - Training on {len(X_train)} samples, validating on {len(val_data[0])} samples.")
        history = self.model.fit(X_train, y_train, batch_size=self.config.BATCH_SIZE, epochs=self.config.EPOCHS, validation_data=val_data, callbacks=callbacks, verbose=1)
        return history
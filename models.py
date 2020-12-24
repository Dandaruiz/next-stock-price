from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
from keras import optimizers


def ltsm(historical_data_train, y_train, historical_data_normalised, historical_data_test, technical_indicators, tech_ind_train, tech_ind_test, y_normaliser, history_points):
	# define two sets of inputs
	lstm_input = Input(shape=(history_points, 7), name='lstm_input')
	dense_input = Input(shape=(technical_indicators.shape[1],), name='tech_input')

	# the first branch operates on the first input
	x = LSTM(50, name='lstm_0')(lstm_input)
	x = Dropout(0.2, name='lstm_dropout_0')(x)
	lstm_branch = Model(inputs=lstm_input, outputs=x)

	# the second branch opreates on the second input
	y = Dense(20, name='tech_dense_0')(dense_input)
	y = Activation("relu", name='tech_relu_0')(y)
	y = Dropout(0.2, name='tech_dropout_0')(y)
	technical_indicators_branch = Model(inputs=dense_input, outputs=y)

	# combine the output of the two branches
	combined = concatenate([lstm_branch.output, technical_indicators_branch.output], name='concatenate')

	z = Dense(64, activation="sigmoid", name='dense_pooling')(combined)
	z = Dense(1, activation="linear", name='dense_out')(z)

	# our model will accept the inputs of the two branches and
	# then output a single value
	model = Model(inputs=[lstm_branch.input, technical_indicators_branch.input], outputs=z)
	adam = optimizers.Adam(lr=0.0005)
	model.compile(optimizer=adam, loss='mse')
	model.fit(x=[historical_data_train, tech_ind_train], y=y_train, batch_size=32, epochs=50, shuffle=True, validation_split=0.1)

	# evaluation
	y_test_predicted = model.predict([historical_data_test, tech_ind_test])
	y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)
	y_predicted = model.predict([historical_data_normalised, technical_indicators])
	y_predicted = y_normaliser.inverse_transform(y_predicted)

	return y_test_predicted, y_predicted
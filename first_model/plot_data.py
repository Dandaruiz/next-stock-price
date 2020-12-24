import matplotlib.pyplot as plt

def plot_forecast(unscaled_y_test, y_test_predicted):
	plt.gcf().set_size_inches(12, 8, forward=True)

	start = 0
	end = -1

	real = plt.plot(unscaled_y_test[start:end], label='real')
	pred = plt.plot(y_test_predicted[start:end], label='predicted')


	plt.legend(['Real', 'Predicted'])

	plt.show()
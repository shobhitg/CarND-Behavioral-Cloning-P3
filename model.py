import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2

from keras.models import Sequential
from keras.layers import Activation, Conv2D, Dense, Dropout, ELU, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import *

from PIL import Image
from sklearn.model_selection import train_test_split
from IPython.display import display
from datetime import datetime

# %matplotlib inline
plt.style.use('ggplot')

# Location of the simulator data.
CSV_DATA_FILE = 'driving_log.csv'

class BehavioralCloningNN:
	def __init__(self, data_path):
		self.data_path = data_path

	def load_csv(self):
		# Load the training data from the simulator.
		cols = ['Center Image', 'Left Image', 'Right Image', 'Steering Angle', 'Throttle', 'Break', 'Speed']
		self.data = pd.read_csv(self.data_path+CSV_DATA_FILE, names=cols, header=1)


		# Uncomment below to trim down the data for quicker testing.
		# self.data = self.data.sample(n=50)

		# Create a timestamp comumn in the data, might get useful later
		df = self.data['Center Image'].str.split('[_.]', expand=True).get([1,2,3,4,5,6,7]).astype(int)
		df.rename(columns={1: 'Year', 2: 'Month', 3: 'Date', 4: 'Hour', 5: 'Min', 6: 'Sec', 7: 'Ms'}, inplace=True)
		df_timestamp = df[['Year', 'Month', 'Date', 'Hour', 'Min', 'Sec', 'Ms']].apply(lambda s : datetime(*s),axis = 1)
		df = self.data
		df.insert(0, 'Timestamp', df_timestamp)
		self.data = df
		self.data.set_index('Timestamp')

		# normalize all numeric fields
		numeric_df = self.data[['Steering Angle', 'Throttle', 'Break', 'Speed']].astype(float)
		data_norm = (numeric_df - numeric_df.mean()) / (numeric_df.max() - numeric_df.min())


		print ("Data csv loaded from folder:", self.data_path)


	# Just for a plot based on timeline.
	def timeline_plot(self):
		mumeric_df = self.data[['Steering Angle', 'Throttle', 'Break', 'Speed']].astype(float)

	# histogram of steering angles
	def hist(self):
		angleHist = self.data['Steering Angle'].hist(bins=50)
		angleHist.set_title('Steering Angle Histogram')

	def preprocessImage(self, image):
		shape = image.shape
		image = image[40:160,:,:]
		image = cv2.resize(image,(200, 66), interpolation=cv2.INTER_AREA)    
		return image

	def load_image(self, path):
		image = cv2.imread((self.data_path + path.strip()))
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		return image

	def load_image_and_preprocess(self, imageData, i_lrc):    
		if (i_lrc == 0):
			path_file = imageData['Left Image']
			shift_ang = .25
		if (i_lrc == 1):
			path_file = imageData['Center Image']
			shift_ang = 0.
		if (i_lrc == 2):
			path_file = imageData['Right Image']
			shift_ang = -.25
		y_steer = imageData['Steering Angle'] + shift_ang
		image = self.load_image(path_file)

		image = self.preprocessImage(image)
		image = np.array(image)
		flipped_image = cv2.flip(image,1)
		flipped_y_steer = -y_steer

		return image,y_steer,flipped_image,flipped_y_steer

	def load_images_and_augment(self):
		indexes = np.arange(len(self.data))

		self.X_train = []
		self.y_train = []

		for i in indexes:
			i_lrc = np.random.randint(3)
			x, y, fx, fy = self.load_image_and_preprocess(self.data.iloc[i], i_lrc)
			self.X_train.append(x)
			self.y_train.append(y)
			self.X_train.append(fx)
			self.y_train.append(fy)

			if (i % 1000 == 0):
				print ('Loaded images:', i)

		print ('Done image loading and augmentation')
		self.X_train = np.array(self.X_train)
		self.y_train = np.array(self.y_train)



	def split_validation(self):
		self.X_train, self.X_valid,self.y_train,self.y_valid = train_test_split(
						self.X_train, 
						self.y_train,
						test_size=0.2,
						random_state=88)

		print(self.X_train.shape, self.y_train.shape, self.X_valid.shape, self.y_valid.shape)

	def make_generators(self):
		self.train_datagen = ImageDataGenerator(
					#rotation_range=5,
					#width_shift_range=0.2,
					shear_range= 0.05,
					zoom_range = 0.05,
					fill_mode = 'nearest'
				  )
		self.train_datagen.fit(self.X_train)
		self.val_datagen = ImageDataGenerator()
		self.val_datagen.fit(self.X_valid)


	def create_model(self):
		self.model = Sequential()
		self.model.add(Convolution2D(24, 5, 5, border_mode='valid', subsample=(2, 2), input_shape=(66, 200, 3), activation="relu"))
		self.model.add(Convolution2D(36, 5, 5, border_mode='valid', subsample=(2, 2), activation="relu"))
		self.model.add(Convolution2D(48, 5, 5, border_mode='valid', subsample=(2, 2), activation="relu"))
		self.model.add(Dropout(0.5))
		self.model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(1, 1), activation="relu"))
		self.model.add(Dropout(0.5))
		self.model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(1, 1), activation="relu"))
		self.model.add(Flatten())
		self.model.add(Dense(1164, activation="relu"))
		self.model.add(Dropout(0.5))
		self.model.add(Dense(100, activation="relu"))
		self.model.add(Dropout(0.5))
		self.model.add(Dense(50, activation="relu"))
		self.model.add(Dense(10, activation="relu"))
		self.model.add(Dense(1))
		self.model.summary()
		# TODO: Compile and train the model
		self.model.compile('adam', 'categorical_crossentropy', ['accuracy'])

	def train(self):
		EPOCH = 30
		opt = Adam(lr=0.001)
		self.model.compile(optimizer=opt, loss='mse', metrics=['accuracy'])
		history = self.model.fit_generator(
			self.train_datagen.flow(self.X_train, self.y_train, batch_size=64), 
			samples_per_epoch=self.X_train.shape[0], 
			nb_epoch=EPOCH,
			validation_data=self.val_datagen.flow(self.X_valid, self.y_valid, batch_size=64), 
			nb_val_samples=self.X_valid.shape[0]
		)
		# list all data in history
		print(history.history.keys())

		data_time_str = datetime.now().strftime("%Y-%m-%d--%H:%M:%S")
		json_string = self.model.to_json()
		json_file = open('model.json', 'w')
		json_file.write(json_string)
		json_file.flush()
		json_file = open('model' + data_time_str + '.json', 'w')
		json_file.write(json_string)
		json_file.flush()

		self.model.save('model.h5')
		self.model.save('model' + data_time_str + '.h5')

		print ('Saved model and a copy of it to model' + data_time_str)



data_path = 'data/'
nn = BehavioralCloningNN(data_path)
nn.load_csv()
nn.load_images_and_augment()
nn.split_validation()
nn.make_generators()
nn.create_model()

nn.train()


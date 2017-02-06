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

		# self.data['Steering Angle'] = pd.rolling_mean(self.data['Steering Angle'], window=5, min_periods=0, center=True)

		# normalize all numeric fields
		numeric_df = self.data[['Steering Angle', 'Throttle', 'Break', 'Speed']].astype(float)
		data_norm = (numeric_df - numeric_df.mean()) / (numeric_df.max() - numeric_df.min())
		# data_norm.plot(figsize=(20, 5))
		#self.data['Steering Angle'] = data_norm['Steering Angle']
		#self.data['Throttle'] = data_norm['Throttle']
		#self.data['Break'] = data_norm['Break']
		#self.data['Speed'] = data_norm['Speed']

		print ("Data csv loaded from folder:", self.data_path)


	# Just for a plot based on timeline.
	def timeline_plot(self):
		mumeric_df = self.data[['Steering Angle', 'Throttle', 'Break', 'Speed']].astype(float)
		# mumeric_df.plot(figsize=(20, 5))

	# histogram of steering angles
	def hist(self):
		angleHist = self.data['Steering Angle'].hist(bins=50)
		angleHist.set_title('Steering Angle Histogram')


	# borrowed from Vivek Yadav's blog
	def augment_brightness_camera_images(self, image):
		image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
		random_bright = .25+np.random.uniform()
		#print(random_bright)
		image1[:,:,2] = image1[:,:,2]*random_bright
		image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
		return image1

	# borrowed from Vivek Yadav's blog
	def trans_image(self, image,steer,trans_range):
		# Translation
		tr_x = trans_range*np.random.uniform()-trans_range/2
		steer_ang = steer + tr_x/trans_range*2*.2
		tr_y = 40*np.random.uniform()-40/2
		#tr_y = 0
		Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
		image_tr = cv2.warpAffine(image,Trans_M,image.shape[0:2][::-1])
		
		return image_tr,steer_ang,Trans_M

	# borrowed from Vivek Yadav's blog
	def add_random_shadow(self, image):
		top_y = 200*np.random.uniform()
		top_x = 0
		bot_x = 66
		bot_y = 200*np.random.uniform()
		image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
		shadow_mask = 0*image_hls[:,:,1]
		X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
		Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
		shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
		#random_bright = .25+.7*np.random.uniform()
		if np.random.randint(2)==1:
			random_bright = .5
			cond1 = shadow_mask==1
			cond0 = shadow_mask==0
			if np.random.randint(2)==1:
				image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
			else:
				image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright    
		image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)
		return image

	def preprocessImage(self, image):
		shape = image.shape
		image = image[40:160,:,:]
		image = cv2.resize(image,(200, 66), interpolation=cv2.INTER_AREA)    
		return image

	def load_image(self, path):
		image = cv2.imread((self.data_path + path.strip()))
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		return image


# images = data[['Center Image', 'Left Image', 'Right Image']]
# angles = data['Steering Angle']
# display (images.head())


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
		# all_images.push(images.iloc[i]['Center Image'])
		# Load the center image and weight.
		image = self.load_image(path_file)

		#image,y_steer,tr_x = self.trans_image(image,y_steer,100)
		#image = self.augment_brightness_camera_images(image)
		#image = self.add_random_shadow(image)
		image = self.preprocessImage(image)
		image = np.array(image)
		#ind_flip = np.random.randint(2)
		#if ind_flip==0:
		flipped_image = cv2.flip(image,1)
		flipped_y_steer = -y_steer

		return image,y_steer,flipped_image,flipped_y_steer



	def load_images_and_augment(self):
		indexes = np.arange(len(self.data))

		self.X_train = []
		self.y_train = []

		for i in indexes:
			# all_images.push(images.iloc[i]['Center Image'])
			# Load the center image and weight.
			# left_image = load_image(self.data.iloc[i]['Left Image'])

			i_lrc = np.random.randint(3)
			x, y, fx, fy = self.load_image_and_preprocess(self.data.iloc[i], i_lrc)
			self.X_train.append(x)
			self.y_train.append(y)
			self.X_train.append(fx)
			self.y_train.append(fy)


			#x, y, fx, fy = self.load_image_and_preprocess(self.data.iloc[i], 0)
			#self.X_train.append(x)
			#self.y_train.append(y)
			#self.X_train.append(fx)
			#self.y_train.append(fy)
			

			#x, y, fx, fy = self.load_image_and_preprocess(self.data.iloc[i], 1)
			#self.X_train.append(x)
			#self.y_train.append(y)
			#self.X_train.append(fx)
			#self.y_train.append(fy)
			

			#x, y, fx, fy = self.load_image_and_preprocess(self.data.iloc[i], 2)
			#self.X_train.append(x)
			#self.y_train.append(y)
			#self.X_train.append(fx)
			#self.y_train.append(fy)
			
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

#X_train, y_train = shuffle(X_train, y_train)
		print(self.X_train.shape, self.y_train.shape, self.X_valid.shape, self.y_valid.shape)

#train_datagen = ImageDataGenerator(
#            )

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
		#self.model.summary()
		# TODO: Compile and train the model
		self.model.compile('adam', 'categorical_crossentropy', ['accuracy'])




	def train(self):
		EPOCH = 30
		opt = Adam(lr=0.001)#, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
		self.model.compile(optimizer=opt, loss='mse', metrics=['accuracy'])
		history = self.model.fit_generator(
		# ==== Unmask below line to dump image out to take snapshot of what's being fed into training process.
			# train_datagen.flow(X_train, y_train, batch_size=128,save_to_dir="./fitgen", save_prefix="img_", save_format="png"), 
		# ==== Use below line to do normal training
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



#data_path = 'data/'
data_path = 'data/'
nn = BehavioralCloningNN(data_path)
nn.load_csv()
nn.load_images_and_augment()
nn.split_validation()
nn.make_generators()
nn.create_model()
nn.train()


from keras.models import Model
from keras.layers import Flatten, Dense
from keras.applications.vgg16 import VGG16
from glob import glob
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

IMAGESHAPE = [224, 224, 3]  # Provide image size as 224 x 224, fixed-size for VGG16 architecture 
vgg_model = VGG16(input_shape=IMAGESHAPE, weights='imagenet', include_top=False) 

# Set layers as non-trainable
for each_layer in vgg_model.layers: 
    each_layer.trainable = False 

# Find number of classes in the train dataset
classes = glob(r'C:\Users\elz00\OneDrive\Desktop\vs code\data\train/*') 

# Define the model up to the first convolutional layer
conv_output_model = Model(inputs=vgg_model.input, outputs=vgg_model.layers[1].output)
conv_output_model.summary()

# Print output shape after flattening
flatten_layer = Flatten()(vgg_model.output) 
print("Output shape after flattening:", flatten_layer.shape)

# Define the prediction layer
prediction = Dense(len(classes), activation='softmax')(flatten_layer) 
final_model = Model(inputs=vgg_model.input, outputs=prediction) 
final_model.summary() 

# Compile the model
final_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print("Optimizer:", final_model.optimizer)  
print("Loss function:", final_model.loss)   
print("Metrics:", final_model.metrics)      

# Image data generator for training data
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

# Print data preprocessing for training data
print("Data preprocessing for training data:")
training_set = train_datagen.flow_from_directory(r'C:\Users\elz00\OneDrive\Desktop\vs code\data\train',
                                                 target_size=(224, 224),
                                                 batch_size=4,
                                                 class_mode='categorical')

for i, batch in enumerate(training_set):
    if i >= 1:  # Print only the first batch
        break
    print("Batch", i+1)
    images, labels = batch
    for j in range(len(images)):
        print("Image", j+1)
        print("Label:", labels[j])
        print("Shape:", images[j].shape)
        
        # Visualize the output of the first convolutional layer
        conv_output = conv_output_model.predict(images[j].reshape(1, 224, 224, 3))
        plt.figure(figsize=(10, 10))
        for k in range(conv_output.shape[-1]):
            plt.subplot(8, 8, k+1)
            plt.imshow(conv_output[0, :, :, k], cmap='viridis')
            plt.axis('off')
        plt.show()
        
        # Visualize the output of the activation function
        activation_output = final_model.predict(images[j].reshape(1, 224, 224, 3))
        print("Activation Output:", activation_output)
        
# Image data generator for test data
test_datagen = ImageDataGenerator(rescale=1. / 255)

# Flow from directory for testing set
test_set = test_datagen.flow_from_directory(r'C:\Users\elz00\OneDrive\Desktop\vs code\data\test',
                                            target_size=(224, 224),
                                            batch_size=4,
                                            class_mode='categorical')

# Fit the model with generator
epochs = 5
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    final_model.fit(training_set,
                    validation_data=test_set,
                    steps_per_epoch=len(training_set),  
                    validation_steps=len(test_set))

# Save the model
final_model.save('model2.keras')
print("The final model is saved.")


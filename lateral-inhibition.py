#Authors: firben-sass & sarphiv

#%%
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import cv2

### SECTION 1 ####################################
#%%
def stimulus_activation(stimulus, weight):
    #Initialize activation array
    activation = np.zeros_like(stimulus, dtype = np.float)

    #Convolve on stimulus
    for i in range(1, len(stimulus) - 1):
        activation[i] = stimulus[i] - weight * (stimulus[i - 1] + stimulus[i + 1])

    #Ignore outermost neuron activations
    return activation[1:-1]


def threshold_activation(activation, threshold):
    return np.array(activation >= threshold, dtype = np.float)


def show_activation(activation):
    plt.bar(np.arange(len(activation)), activation)
    plt.show()
    print(f"Activation levels: {activation}")


stimulus = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])


#Raw activation
activation = stimulus_activation(stimulus, 0.1)
#Edge detection
edges = threshold_activation(activation, 0.9)

#Show raw and edge activations
show_activation(activation)
show_activation(edges)


### SECTION 2 ####################################
#%%
stimulus = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5])

activation = stimulus_activation(stimulus, 0.1)

#Mach bands illusion
show_activation(activation)


### SECTION 3 ####################################
#%%
def stimulus_activation(stimulus, weight):
    kernel = np.array([-weight, 1, -weight])

    activation = np.convolve(stimulus, kernel, mode='same')

    return activation[1:-1]


### SECTION 4 ####################################
#%%
def stimulus_2d_activation(stimulus, weight, focus_size):
    #Initialize kernel with weight everywhere
    kernel_size = focus_size * 3
    kernel = -weight * np.ones((kernel_size,) * 2)

    #Initialize center focus area
    focus = np.ones((focus_size,) * 2)

    #Overwrite center of kernel with focus area
    kernel[focus_size:-focus_size, focus_size:-focus_size] = focus

    #Convolve on stimulus using kernel
    activation = signal.convolve2d(stimulus, kernel, mode='same')


    #Ignore outermost neuron activations
    activation = activation[focus_size:-focus_size, 
                            focus_size:-focus_size]

    #Normalize activation levels
    a_max, a_min = activation.max(), activation.min()
    activation = (activation - a_min) / (a_max - a_min)

    #Return post-processed activation
    return activation


def show_2d_activation(activation):
    plt.imshow(activation, cmap = "gray")
    plt.axis("off")
    plt.show()


def load_image(path):
    #Load and return grey scale image
    return cv2.imread(path).mean(axis = 2)


#%%
#Load Mona Lisa image
img_mona = load_image("./MonaLisa.jpg")

#Display unmodified Mona Lisa image
show_2d_activation(img_mona)

#%%
#Convolve stimulus
img_mona_activated = stimulus_2d_activation(img_mona, 0.1, 16)
show_2d_activation(img_mona_activated)


#%%
#Edge detect stimulus face
img_mona_face_edges_activated = stimulus_2d_activation(img_mona, 0.1, 10)
img_mona_face_edges = threshold_activation(img_mona_face_edges_activated, 0.5)
show_2d_activation(img_mona_face_edges)

#%%
#Edge detect stimulus all
img_mona_all_edges_activated = stimulus_2d_activation(img_mona, 0.1, 16)
img_mona_all_edges = threshold_activation(img_mona_all_edges_activated, 0.4)
show_2d_activation(img_mona_all_edges)


### SECTION 5 ####################################
#%%
#Load Hermann grid
img_hermann = load_image("./hermann.jpg")

#Display unmodified Hermann grid
show_2d_activation(img_hermann)

#%%
#Convolve stimulus
img_hermann_activated = stimulus_2d_activation(img_hermann, 0.1, 26)
show_2d_activation(img_hermann_activated)

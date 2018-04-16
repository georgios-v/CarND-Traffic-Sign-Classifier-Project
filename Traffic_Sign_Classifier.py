
# coding: utf-8

# # Self-Driving Car Engineer Nanodegree
# 
# ## Deep Learning
# 
# ## Project: Build a Traffic Sign Recognition Classifier
# 
# In this notebook, a template is provided for you to implement your functionality in stages, which is required to successfully complete this project. If additional code is required that cannot be included in the notebook, be sure that the Python code is successfully imported and included in your submission if necessary. 
# 
# > **Note**: Once you have completed all of the code implementations, you need to finalize your work by exporting the iPython Notebook as an HTML document. Before exporting the notebook to html, all of the code cells need to have been run so that reviewers can see the final implementation and output. You can then export the notebook by using the menu above and navigating to  \n",
#     "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission. 
# 
# In addition to implementing code, there is a writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a [write up template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) that can be used to guide the writing process. Completing the code template and writeup template will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/481/view) for this project.
# 
# The [rubric](https://review.udacity.com/#!/rubrics/481/view) contains "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. The stand out suggestions are optional. If you decide to pursue the "stand out suggestions", you can include the code in this Ipython notebook and also discuss the results in the writeup file.
# 
# 
# >**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

# ---
# ## Step 0: Load The Data

# In[1]:


import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import pprint
import tensorflow as tf

from numpy import random as rnd
from sklearn.utils import shuffle
from tensorflow.contrib.layers import flatten

get_ipython().run_line_magic('matplotlib', 'inline')
pprinter = pprint.PrettyPrinter(indent=4)
pp = pprinter.pprint


# In[2]:


# Load pickled data
training_file = "../traffic-signs-data/train.p"
validation_file = "../traffic-signs-data/valid.p"
testing_file = "../traffic-signs-data/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X = {'train': train['features'], 'valid': valid['features'], 'test': test['features']}
Y = {'train': train['labels'], 'valid': valid['labels'], 'test': test['labels']}


# ---
# 
# ## Step 1: Dataset Summary & Exploration
# 
# The pickled data is a dictionary with 4 key/value pairs:
# 
# - `'features'` is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
# - `'labels'` is a 1D array containing the label/class id of the traffic sign. The file `signnames.csv` contains id -> name mappings for each id.
# - `'sizes'` is a list containing tuples, (width, height) representing the original width and height the image.
# - `'coords'` is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. **THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES**
# 
# Complete the basic data summary below. Use python, numpy and/or pandas methods to calculate the data summary rather than hard coding the results. For example, the [pandas shape method](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.shape.html) might be useful for calculating some of the summary results. 

# ### Provide a Basic Summary of the Data Set Using Python, Numpy and/or Pandas

# In[3]:


### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
n_train = len(X['train'])

# TODO: Number of validation examples
n_validation = len(X['valid'])

# TODO: Number of testing examples.
n_test = len(X['test'])

# TODO: What's the shape of an traffic sign image?
image_shape = X['train'][0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(np.unique(Y['train']))

print("Number of training examples =", n_train)
print("Number of validation examples =", n_validation)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)


# ### Include an exploratory visualization of the dataset

# Visualize the German Traffic Signs Dataset using the pickled file(s). This is open ended, suggestions include: plotting traffic sign images, plotting the count of each sign, etc. 
# 
# The [Matplotlib](http://matplotlib.org/) [examples](http://matplotlib.org/examples/index.html) and [gallery](http://matplotlib.org/gallery.html) pages are a great resource for doing visualizations in Python.
# 
# **NOTE:** It's recommended you start with something simple first. If you wish to do more, come back to it after you've completed the rest of the sections. It can be interesting to look at the distribution of classes in the training, validation and test set. Is the distribution the same? Are there more examples of some classes than others?

# In[4]:


# Load sign names file
db = pd.read_csv("signnames.csv")
db.set_index("ClassId")
db.head(n=5)


# In[5]:


### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
# Visualizations will be shown in the notebook.
def show_image(img):
    plt.figure(figsize=(1,1))
    plt.imshow(img)


def show_rnd_image(X):
    index = rnd.randint(0, len(X))
    image = X[index].squeeze()
    show_image(image)
    return index


print("Label:", Y['train'][show_rnd_image(X['train'])])


# #### Class imbalance

# In[6]:


# frequency analysis
def freq(x):
    y = np.bincount(x)
    return y


def group_by_freq(y, db):
    f = freq(y)
    arr = []
    for i, v in enumerate(f):
        arr.append({"name": db.loc[[i]]["SignName"].values[0], "freq": v})
    return pd.DataFrame(arr)


def plot_freq(y):
    y.plot(kind='bar', x='name', figsize=(16, 4), sort_columns=True)


# In[7]:


Y_freq = {}
Y_freq['train'] = group_by_freq(Y['train'], db)
plot_freq(Y_freq['train'])
Y_freq['valid'] = group_by_freq(Y['valid'], db)
plot_freq(Y_freq['valid'])
Y_freq['test'] = group_by_freq(Y['test'], db)
plot_freq(Y_freq['test'])


# In[8]:


def group_by_lbl(y):
    arr = {}
    for i, l in enumerate(y):
        if l not in arr:
            arr[l] = []
        arr[l].append(i)
    return arr


# group Y by label: {'train': {label: [images]}, ...}
def create_y_gbl(y):
    y_gbl = {}
    for lbl, images in y.items():
        y_gbl[lbl] = group_by_lbl(y[lbl])
    return y_gbl


Y_gbl = create_y_gbl(Y)


# #### Data sample visualization

# In[9]:


def get_sample(S, y, label, n):
    for lbl, images in y.items():
        if lbl not in S:
            S[lbl] = {'train': [], 'valid': [], 'test': []}
        s = list(rnd.choice(images, n))
        S[lbl][label].extend(s)


def create_sample(y, sz=[3, 2, 2]):
    sample = {}
    get_sample(sample, y['train'], 'train', sz[0])
    get_sample(sample, y['valid'], 'valid', sz[1])
    get_sample(sample, y['test'], 'test', sz[2])
    return sample


def show_image_list(img_list, title, subtitle=None, rows=1, figsize=(14, 1.6)):
    count = len(img_list)
    cols = int(np.ceil(count/rows))
    if subtitle is None:
        subtitle = [""]*count
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=figsize)
    if rows == 1:
        axes = [axes]
    cmap = None
    for row in range(rows):
        for col in range(cols):
            i = row*cols + col
            img = img_list[i]
            if len(img.shape) < 3 or img.shape[-1] < 3:
                cmap = "gray"
                img = np.reshape(img, (img.shape[0], img.shape[1]))
            axes[row][col].axis("off")
            axes[row][col].set_title(subtitle[i])
            axes[row][col].imshow(img, cmap=cmap)
    fig.suptitle(title, fontsize=12, fontweight='bold', y=1)
    fig.tight_layout()
    plt.show()


def show_sample(x, sample, limit=None):
    # visualize the sample
    sample_ = sample
    if limit is not None:
        sample_ = {k:sample[k] for k in rnd.choice(list(sample.keys()), limit) if k in sample}
    for lbl, sets in sample_.items():
        img_list = []
        for s, images in sets.items():
            for i in images:
                img_list.append(x[s][i])
        show_image_list(img_list, "%s: %s" % (lbl, db.loc[[lbl]]["SignName"].values[0]))


# In[10]:


Y_sample = create_sample(Y_gbl)
show_sample(X, Y_sample, limit=1)


# ----
# 
# ## Step 2: Design and Test a Model Architecture
# 
# Design and implement a deep learning model that learns to recognize traffic signs. Train and test your model on the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).
# 
# The LeNet-5 implementation shown in the [classroom](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) at the end of the CNN lesson is a solid starting point. You'll have to change the number of classes and possibly the preprocessing, but aside from that it's plug and play! 
# 
# With the LeNet-5 solution from the lecture, you should expect a validation set accuracy of about 0.89. To meet specifications, the validation set accuracy will need to be at least 0.93. It is possible to get an even higher accuracy, but 0.93 is the minimum for a successful project submission. 
# 
# There are various aspects to consider when thinking about this problem:
# 
# - Neural network architecture (is the network over or underfitting?)
# - Play around preprocessing techniques (normalization, rgb to grayscale, etc)
# - Number of examples per label (some have more than others).
# - Generate fake data.
# 
# Here is an example of a [published baseline model on this problem](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). It's not required to be familiar with the approach used in the paper but, it's good practice to try to read papers like these.

# ### Pre-process the Data Set (normalization, grayscale, etc.)

# Minimally, the image data should be normalized so that the data has mean zero and equal variance. For image data, `(pixel - 128)/ 128` is a quick way to approximately normalize the data and can be used in this project. 
# 
# Other pre-processing steps are optional. You can try different techniques to see if it improves performance. 
# 
# Use the code cell (or multiple code cells, if necessary) to implement the first step of your project.

# In[11]:


## ON/OFF Switches for the various preprocessing options available
GRAYSCALE = False
SHUFFLE = True
GENERATION = False
BRIGHTNESS = True
YUV = True
NORMALIZATION = False
SHOW_SAMPLE = True
SAMPLE_SIZE = 5


# #### Grayscale

# In[12]:


def apply_grayscale(x):
    x_gs = {}
    for lbl, images in x.items():
        x_gs[lbl] = []
        for img in images:
            [in_h, in_w, in_d] = img.shape
            img_gs = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img_gs = np.reshape(img_gs, (in_h, in_w, 1))
            x_gs[lbl].append(img_gs)
    return x_gs


# #### Shuffle

# In[13]:


def apply_shuffle(x, y):
    return shuffle(x, y)


# #### Sample generation

# To address the vast imbalance between labels in the training set, new samples will be generated

# In[14]:


def transform_image(img, ang_range, shear_range, trans_range):
    '''
    This function transforms images to generate new images.
    The function takes in following arguments,
    1- Image
    2- ang_range: Range of angles for rotation
    3- shear_range: Range of values to apply affine transform to
    4- trans_range: Range of values to apply translations over.

    A Random uniform distribution is used to generate different parameters for transformation

    '''
    # Rotation
    ang_rot = np.random.uniform(ang_range) - ang_range / 2
    rows, cols, ch = img.shape    
    Rot_M = cv2.getRotationMatrix2D((cols / 2, rows / 2), ang_rot, 1)

    # Translation
    tr_x = trans_range * np.random.uniform() - trans_range / 2
    tr_y = trans_range * np.random.uniform() - trans_range / 2
    Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])

    # Shear
    pts1 = np.float32([[5, 5], [20, 5], [5, 20]])
    pt1 = 5 + shear_range * np.random.uniform() - shear_range / 2
    pt2 = 20 + shear_range * np.random.uniform() - shear_range / 2

    # Brightness
    pts2 = np.float32([[pt1, 5], [pt2, pt1], [5, pt2]])
    shear_M = cv2.getAffineTransform(pts1, pts2)
    img = cv2.warpAffine(img, Rot_M, (cols, rows))
    img = cv2.warpAffine(img, Trans_M, (cols, rows))
    img = cv2.warpAffine(img, shear_M, (cols, rows))
    
    return img


def apply_generation(x, y, y_frq=None, y_gbl=None, plot=True):
    if y_frq is None:
        y_frq = group_by_freq(y, db)
    if y_gbl is None:
        y_gbl = group_by_lbl(y)
    if plot:
        print("Current class frequency")
        plot_freq(y_frq)
    x_new = []
    y_new = []
    max_freq = y_frq.loc[y_frq['freq'].idxmax()]['freq']
    for lbl, images in y_gbl.items():
        freq = y_frq.loc[lbl]['freq']
        diff = max_freq - freq
        per_image = int(diff / len(images) + 0.5)  # manual round
        for i in images:
            for j in range(per_image):
                img = x[i]
                x_new.append(transform_image(img, 20, 10, 5))
                y_new.append(lbl)
    x_new, y_new = apply_shuffle(x_new, y_new)
    x = np.append(x, x_new, axis=0)
    y = np.append(y, y_new, axis=0)
    if plot:
        print("New class frequency")
        plot_freq(group_by_freq(y, db))
    return x, y


# #### Brightness

# In[15]:


def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def apply_brightness(x):
    x_brt = {}
    for lbl, images in x.items():
        x_brt[lbl] = []
        for img in images:
            x_brt[lbl].append(increase_brightness(img))
    return x_brt


# #### Image channel processing (YUV)

# Convert images into YUV format, as it is documented to improve accuracy

# In[16]:


def apply_yuv(x):
    x_yuv = {}
    for lbl, images in x.items():
        x_yuv[lbl] = []
        for img in images:
            img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            x_yuv[lbl].append(img_yuv)
    return x_yuv


# #### Normalization

# In[17]:


def normalise_images(img, dist):
    std = np.std(dist)
    mean = np.mean(dist)
    return (img - mean) / std

def apply_normalization(x):
    x_norm = {}
    for lbl, images in x.items():
        x_norm[lbl] = normalise_images(images, images)
    return x_norm


# In[18]:


def preprocess(x, y, gs=GRAYSCALE, gen=GENERATION, sfl=SHUFFLE, bright=BRIGHTNESS, yuv=YUV, norm=NORMALIZATION, show=SHOW_SAMPLE*SAMPLE_SIZE):
    global Y_gbl, Y_sample, n_train
    if gen and 'train' in x:
        print("Generating new images...")
        x['train'], y['train'] = apply_generation(x['train'], y['train'], y_frq=Y_freq['train'], y_gbl=Y_gbl['train'], plot=show)
        if show:
            print("Sample new images")
            Y_gbl['train'] = group_by_lbl(y['train'][n_train:])
            Y_sample = create_sample(Y_gbl)
            show_sample(x, Y_sample, limit=show)
        n_train = len(x['train'])
    if bright:
        print("Improving brightness...")
        x = apply_brightness(x)
        if show:
            show_sample(x, Y_sample, limit=show)
    if yuv:
        print("Converting to YUV...")
        x = apply_yuv(x)
        if show:
            show_sample(x, Y_sample, limit=show)
    if gs:
        print("Converting to grayscale")
        x = apply_grayscale(x)
        if show:
            show_sample(x, Y_sample, limit=show)
    if sfl and 'train' in x:
        print("Shuffling train dataset...")
        x['train'], y['train'] = apply_shuffle(x['train'], y['train'])
        Y_gbl = create_y_gbl(y)
        Y_sample = create_sample(Y_gbl)
        if show:
            show_sample(x, Y_sample, limit=show)
    if norm:
        print("Normalizing images...")
        x = apply_normalization(x)
        if show:
            show_sample(x, Y_sample, limit=show)
    print('Done')
    return x,y


# In[19]:


X, Y = preprocess(X, Y)


# ### Model Architecture

# #### Model building blocks

# In[20]:


image_shape = X['train'][0].shape
if len(image_shape) < 3:
    x = tf.placeholder(tf.float32, (None, image_shape[0], image_shape[1]))
else:
    x = tf.placeholder(tf.float32, (None, image_shape[0], image_shape[1], image_shape[2]))
y = tf.placeholder(tf.int32, (None))
keep_prob = tf.placeholder(tf.float32)
one_hot_y = tf.one_hot(y, n_classes)


# In[21]:


### Define your architecture here.
### Feel free to use as many code cells as needed.

def get_F(out_h, in_h, S):
    # print(out_h, in_h, S)
    return in_h + 1 - (out_h * S)


def conv2d(x, out_w, out_h, out_d, mu, sigma):
    [in_batch, in_h, in_w, in_d] = x.get_shape().as_list()
    S = 1
    out_shape = [get_F(out_w, in_w, S), get_F(out_h, in_h, S), in_d, out_d]
    # print(out_shape)
    F_W = tf.Variable(tf.truncated_normal(out_shape, mean=mu, stddev=sigma))
    F_b = tf.Variable(tf.zeros(out_d))
    strides = [1, S, S, 1]
    padding = 'VALID'
    return tf.nn.conv2d(x, F_W, strides, padding) + F_b


def maxpool2d(x, out):
    [in_batch, in_h, in_w, in_d] = x.get_shape().as_list()
    S = 2
    k = get_F(out, in_w, S)
    # print(k)
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, S, S, 1], padding='VALID')


def fully_connd(x, out_sz, mu, sigma):
    [in_batch, in_sz] = x.get_shape().as_list()
    weights = tf.Variable(tf.truncated_normal([in_sz, out_sz], mean=mu, stddev=sigma))
    biases = tf.Variable(tf.zeros(out_sz))
    return tf.add(tf.matmul(x, weights), biases)
    

def LeNet(x, dropout):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    cnv_activ = tf.nn.lrn
    flc_activ = tf.nn.relu
    
    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1 = conv2d(x, 28, 28, 6, mu, sigma)
    print("conv1:", conv1)
    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = maxpool2d(conv1, 14)
    print("pool1:", conv1)
    # TODO: Activation.
    conv1 = cnv_activ(conv1)
    print("actv1:", conv1)
    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    conv2 = conv2d(conv1, 10, 10, 16, mu, sigma)
    print("conv2:", conv2)
    # TODO: Activation.
    conv2 = cnv_activ(conv2)
    print("actv2:", conv2)
    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = maxpool2d(conv2, 5)
    print("pool2:", conv2)
    # TODO: Flatten. Input = 5x5x16. Output = 400.
    fc0 = flatten(conv2)
    print("flatten:", fc0)
    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1 = fully_connd(fc0, 120, mu, sigma)
    print("fully1:", fc1)
    # TODO: Activation.
    fc1 = flc_activ(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)
    print("actv3:", fc1)
    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2 = fully_connd(fc1, 84, mu, sigma)
    print("fully2:", fc2)
    # TODO: Activation.
    fc2 = flc_activ(fc2)
    fc2 = tf.nn.dropout(fc2, dropout)
    print("actv4:", fc2)
    # TODO: Layer 5: Fully Connected. Input = 84. Output = 10.
    logits = fully_connd(fc2, n_classes, mu, sigma)
    print("fully3:", logits)
    # logits = tf.nn.softmax(logits)
    # print("softmax:", logits)
    return logits, conv1, conv2, fc1, fc2


# #### Build the model

# In[22]:


### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.
rate = 0.001

logits, conv1, conv2, fc1, fc2 = LeNet(x, keep_prob)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)


# #### Build the Evaluation

# In[23]:


prediction_operation = tf.argmax(logits, 1)
correct_prediction = tf.equal(prediction_operation, tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


# ### Train, Validate and Test the Model

# A validation set can be used to assess how well the model is performing. A low accuracy on the training and validation
# sets imply underfitting. A high accuracy on the training set but low accuracy on the validation set implies overfitting.

# #### Parameters and encoding

# In[24]:


EPOCHS = 10
BATCH_SIZE = 100
dropout = 0.7


# #### Train and Evaluation

# In[25]:


with tf.Session() as sess:
    try:
        saver.restore(sess, './lenet')
        print('loaded model')
    except:
        sess.run(tf.global_variables_initializer())
        print('initialized new model')
    num_examples = len(X['train'])
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X['train'], Y['train'] = shuffle(X['train'], Y['train'])
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X['train'][offset:end], Y['train'][offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
            
        train_accuracy = evaluate(X['train'], Y['train'])
        validation_accuracy = evaluate(X['valid'], Y['valid'])
        print("EPOCH {} ...".format(i+1))
        print("Training Accuracy = {:.3f}".format(train_accuracy))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
    
    saver.save(sess, './lenet')
    print("Model saved")


# #### Test

# In[26]:


with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X['test'], Y['test'])
    print("Test Accuracy = {:.3f}".format(test_accuracy))


# ---
# 
# ## Step 3: Test a Model on New Images
# 
# To give yourself more insight into how your model is working, download at least five pictures of German traffic signs from the web and use your model to predict the traffic sign type.
# 
# You may find `signnames.csv` useful as it contains mappings from the class id (integer) to the actual sign name.

# ### Load and Output the Images

# In[27]:


### Load the images and plot them here.
### Feel free to use as many code cells as needed.

path_new = "./traffic-signs-custom/"
db_new = pd.read_csv(path_new + "images.csv")
db_new.set_index("ClassId")

X_new = {'test':[]}
Y_new = {'test':[]}
subtitles = []
for r in db_new.itertuples():
    image = mpimg.imread(path_new + r[2])
    X_new['test'].append(image)
    Y_new['test'].append(r[1])
    subtitles.append("%s: %s" % (r[1], db.loc[[r[1]]]["SignName"].values[0]))
show_image_list(X_new['test'], 'New Images', subtitle=subtitles, rows=2, figsize=(14, 3.8))


# ### Predict the Sign Type for Each Image

# In[28]:


### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
### Feel free to use as many code cells as needed.
X_new, Y_new = preprocess(X_new, Y_new, show=False)
show_image_list(X_new['test'], 'New Images', subtitle=subtitles, rows=2, figsize=(14, 3.8))


# In[29]:


with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    batch_x, batch_y = X_new['test'][:], Y_new['test'][:]
    prediction_prob = sess.run(logits, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
    prediction = sess.run(tf.argmax(logits, 1), feed_dict={logits:prediction_prob})
subtitles = []
for i, p in enumerate(prediction):
    subtitles.append("%s: %s (%s)" % (p, db.loc[[p]]["SignName"].values[0], p == Y_new['test'][i]))
show_image_list(X_new['test'], 'Predictions', subtitle=subtitles, rows=2, figsize=(14, 3.8))


# ### Analyze Performance

# In[30]:


### Calculate the accuracy for these 5 new images. 
### For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate on these new images.
t = len(prediction)
accuracy = sum([p == Y_new['test'][i] for i, p in enumerate(prediction)])
print("Accuracy:", accuracy / t, 'or', "%s/%s" %(accuracy, t))


# ### Output Top 5 Softmax Probabilities For Each Image Found on the Web

# For each of the new images, print out the model's softmax probabilities to show the **certainty** of the model's predictions (limit the output to the top 5 probabilities for each image). [`tf.nn.top_k`](https://www.tensorflow.org/versions/r0.12/api_docs/python/nn.html#top_k) could prove helpful here. 
# 
# The example below demonstrates how tf.nn.top_k can be used to find the top k predictions for each image.
# 
# `tf.nn.top_k` will return the values and indices (class ids) of the top k predictions. So if k=3, for each sign, it'll return the 3 largest probabilities (out of a possible 43) and the correspoding class ids.
# 
# Take this numpy array as an example. The values in the array represent predictions. The array contains softmax probabilities for five candidate images with six possible classes. `tf.nn.top_k` is used to choose the three classes with the highest probability:
# 
# ```
# # (5, 6) array
# a = np.array([[ 0.24879643,  0.07032244,  0.12641572,  0.34763842,  0.07893497,
#          0.12789202],
#        [ 0.28086119,  0.27569815,  0.08594638,  0.0178669 ,  0.18063401,
#          0.15899337],
#        [ 0.26076848,  0.23664738,  0.08020603,  0.07001922,  0.1134371 ,
#          0.23892179],
#        [ 0.11943333,  0.29198961,  0.02605103,  0.26234032,  0.1351348 ,
#          0.16505091],
#        [ 0.09561176,  0.34396535,  0.0643941 ,  0.16240774,  0.24206137,
#          0.09155967]])
# ```
# 
# Running it through `sess.run(tf.nn.top_k(tf.constant(a), k=3))` produces:
# 
# ```
# TopKV2(values=array([[ 0.34763842,  0.24879643,  0.12789202],
#        [ 0.28086119,  0.27569815,  0.18063401],
#        [ 0.26076848,  0.23892179,  0.23664738],
#        [ 0.29198961,  0.26234032,  0.16505091],
#        [ 0.34396535,  0.24206137,  0.16240774]]), indices=array([[3, 0, 5],
#        [0, 1, 4],
#        [0, 5, 1],
#        [1, 3, 5],
#        [1, 4, 3]], dtype=int32))
# ```
# 
# Looking just at the first row we get `[ 0.34763842,  0.24879643,  0.12789202]`, you can confirm these are the 3 largest probabilities in `a`. You'll also notice `[3, 0, 5]` are the corresponding indices.

# In[31]:


### Print out the top five softmax probabilities for the predictions on the German traffic sign images found on the web. 
### Feel free to use as many code cells as needed.
def print_soft_max(i):
    j = Y_new['test'][i]
    print()
    print("%s: %s" % (j, db.loc[[j]]["SignName"].values[0]))
    show_image(X_new['test'][i])
    plt.show(block=True)
    for j in range(k):
        print('%s: %s (%s)' % (softmax_ind[i][j], db.loc[[softmax_ind[i][j]]]['SignName'].values[0], softmax_prob[i][j]))
    print()


k = 5
with tf.Session() as sess:
    softmax_prob, softmax_ind = sess.run(tf.nn.top_k(tf.constant(prediction_prob), k=k))
for i in range(len(X_new['test'])):
    print_soft_max(i)


# ### Project Writeup
# 
# Once you have completed the code implementation, document your results in a project writeup using this [template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) as a guide. The writeup can be in a markdown or pdf file. 

# > **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  \n",
#     "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.

# ---
# 
# ## Step 4 (Optional): Visualize the Neural Network's State with Test Images
# 
#  This Section is not required to complete but acts as an additional excersise for understaning the output of a neural network's weights. While neural networks can be a great learning device they are often referred to as a black box. We can understand what the weights of a neural network look like better by plotting their feature maps. After successfully training your neural network you can see what it's feature maps look like by plotting the output of the network's weight layers in response to a test stimuli image. From these plotted feature maps, it's possible to see what characteristics of an image the network finds interesting. For a sign, maybe the inner network feature maps react with high activation to the sign's boundary outline or to the contrast in the sign's painted symbol.
# 
#  Provided for you below is the function code that allows you to get the visualization output of any tensorflow weight layer you want. The inputs to the function should be a stimuli image, one used during training or a new one you provided, and then the tensorflow variable name that represents the layer's state during the training process, for instance if you wanted to see what the [LeNet lab's](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) feature maps looked like for it's second convolutional layer you could enter conv2 as the tf_activation variable.
# 
# For an example of what feature map outputs look like, check out NVIDIA's results in their paper [End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) in the section Visualization of internal CNN State. NVIDIA was able to show that their network's inner weights had high activations to road boundary lines by comparing feature maps from an image with a clear path to one without. Try experimenting with a similar test to show that your trained network's weights are looking for interesting features, whether it's looking at differences in feature maps from images with or without a sign, or even what feature maps look like in a trained network vs a completely untrained one on the same sign image.
# 
# <figure>
#  <img src="visualize_cnn.png" width="380" alt="Combined Image" />
#  <figcaption>
#  <p></p> 
#  <p style="text-align: center;"> Your output should look something like this (above)</p> 
#  </figcaption>
# </figure>
#  <p></p> 
# 

# In[60]:


### Visualize your network's feature maps here.
### Feel free to use as many code cells as needed.

# image_input: the test image being fed into the network to produce the feature maps
# tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
# activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and max values of the output
# plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry

def outputConvFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1, plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it may be having trouble accessing the variable from inside a function
    cmap = "gray"
    if len(image_input.shape) > 3:
        cmap = None
    activation = tf_activation.eval(session=sess,feed_dict={x : image_input})
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(15,15))
    for featuremap in range(featuremaps):
        plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap=cmap)
        elif activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap=cmap)
        elif activation_min !=-1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap=cmap)
        else:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap=cmap)


def outputFCFeatureMap(image_input, tf_activation, dropout, activation_min=-1, activation_max=-1, plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it may be having trouble accessing the variable from inside a function
    cmap = "gray"
    if len(image_input.shape) > 3:
        cmap = None
    activation = tf_activation.eval(session=sess,feed_dict={x: image_input, keep_prob: dropout})
    plt.imshow(activation, interpolation="nearest", cmap=cmap)


# In[62]:


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(len(X_new['test'])):
        img = X_new['test'][i]
        if len(img.shape) > 2:
            rimg = np.reshape(img, (1, shape[0], shape[1], shape[2]))
        else:
            rimg = np.reshape(img, (1, shape[0], shape[1]))
        y = Y_new['test'][i]
        print(y, db.loc[y]['SignName'])
        show_image(img)
        plt.show(block=True)
        shape = img.shape
        print("Convolution 1")
        outputConvFeatureMap(rimg, conv1)
        plt.show(block=True)
        print("Convolution 2")
        outputConvFeatureMap(rimg, conv2)
        plt.show(block=True)
        print("Fully-connected 1")
        outputFCFeatureMap(rimg, fc1, dropout)
        plt.show(block=True)
        print("Fully-connected 2")
        outputFCFeatureMap(rimg, fc2, dropout)
        plt.show(block=True)
        print()


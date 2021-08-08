import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import random
import shutil
from random import sample
from datetime import datetime
import cv2
import matplotlib.font_manager as fm
from matplotlib.collections import QuadMesh

# Model 
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam



#Image preprocessing
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
_ROOT = os.getcwd()
# dictionary of paths
all_paths_dic = {'summary_file_path':os.path.join('train','train.csv'),
         'data_train_path':os.path.join('train','train'),
         'test_train_path':get_data('test'),
         'glaucoma_train_path':os.path.join(os.getcwd(),'glaucoma_data','train'),
         'glaucoma_test_path':os.path.join(os.getcwd(),'glaucoma_data','test'),
         'diabetic_ret_train_path':os.path.join(os.getcwd(),'diabetic_ret_data','train'),
         'diabetic_ret_test_path':os.path.join(os.getcwd(),'diabetic_ret_data','test')}

##################################################
#
#             UTILITY FUNCTIONS
#
##################################################

# Access Utility Function 
def get_data(path):
    return os.path.join(_ROOT,'Data', path)

# Utility function to split dataset
def split_train_test(df_orig,train_split = 0.7,paths_dic=all_paths_dic):
  # Method to create train / split and copy all files accordingly to respective folder
  # Args:
    # train_split -> float [0,1] representing train split
    # paths_dic -> dictionary representing all paths
    # df_orig -> original data frame
  # Returns:
    # ---


  def copy_files(current_dir,dist_dir,file_names):
    # helper function to copy files from one directory to another
    for file in file_names:
      src = os.path.join(current_dir,file)
      dst = os.path.join(os.getcwd(),dist_dir,file)
      shutil.copy(src, dst)

  # Getting inx for each positive case
  glaucoma_idx     = (df_orig["glaucoma"]==1).values
  diabetic_ret_idx = (df_orig["diabetic retinopathy"]==1).values
  normal_idx       =  (df_orig["normal"]==1).values

  # Getting train size
  train_size_glaucoma     = int(glaucoma_idx.sum() * train_split)
  train_size_diabetic_ret = int(diabetic_ret_idx.sum() * train_split)
  train_size_normal      = int(normal_idx.sum() * train_split)
  
  random.seed(1)

  # glaucoma file names (train / test)
  glaucoma_train     = random.sample(df_orig.loc[glaucoma_idx,"filename"].values.tolist(),train_size_glaucoma)
  glaucoma_test      = [file for file in df_orig.loc[glaucoma_idx,"filename"].values.tolist() if file not in glaucoma_train]
  # diabetic ret file names (train / test)
  diabetic_ret_train = random.sample(df_orig.loc[diabetic_ret_idx,"filename"].values.tolist(),train_size_diabetic_ret)
  diabetic_ret_test      = [file for file in df_orig.loc[diabetic_ret_idx,"filename"].values.tolist() if file not in diabetic_ret_train]
  # neither file names (train / test)
  normal_train      = random.sample(df_orig.loc[normal_idx,"filename"].values.tolist(),train_size_normal)
  normal_test       = [file for file in df_orig.loc[normal_idx,"filename"].values.tolist() if file not in normal_train]

  # creating all directories
  # creating training directories
  try:
    os.makedirs(os.path.join(all_paths_dic['glaucoma_train_path'],'positive'))
  except:
    print("directory exists!")
  try:
    os.makedirs(os.path.join(all_paths_dic['glaucoma_train_path'],'negative'))
  except:
    print("directory exists!")
  try:
    os.makedirs(os.path.join(all_paths_dic['diabetic_ret_train_path'],'positive'))
  except:
    print("directory exists!")
  try:
    os.makedirs(os.path.join(all_paths_dic['diabetic_ret_train_path'],'negative'))
  except:
    print("directory exists!")

  #creating testing directories
  try:
    os.makedirs(os.path.join(all_paths_dic['glaucoma_test_path'],'positive'))
  except:
    print("directory exists!")
  try:
    os.makedirs(os.path.join(all_paths_dic['glaucoma_test_path'],'negative'))
  except:
    print("directory exists!")
  try:
    os.makedirs(os.path.join(all_paths_dic['diabetic_ret_test_path'],'positive'))
  except:
    print("directory exists!")
  try:
    os.makedirs(os.path.join(all_paths_dic['diabetic_ret_test_path'],'negative'))
  except:
    print("directory exists!")

  # copying files from main directory to glaucome train / test
  main_directory = get_data(all_paths_dic['data_train_path'])
  copy_files(main_directory,os.path.join(all_paths_dic['glaucoma_train_path'],'positive'),glaucoma_train)
  copy_files(main_directory,os.path.join(all_paths_dic['glaucoma_train_path'],'negative'),normal_train)

  copy_files(main_directory,os.path.join(all_paths_dic['glaucoma_test_path'],'positive'),glaucoma_test)
  copy_files(main_directory,os.path.join(all_paths_dic['glaucoma_test_path'],'negative'),normal_test)

  # copying files from main directory to diabetic_ret train / test
  copy_files(main_directory,os.path.join(all_paths_dic['diabetic_ret_train_path'],'positive'),diabetic_ret_train)
  copy_files(main_directory,os.path.join(all_paths_dic['diabetic_ret_train_path'],'negative'),normal_train)

  copy_files(main_directory,os.path.join(all_paths_dic['diabetic_ret_test_path'],'positive'),diabetic_ret_test)
  copy_files(main_directory,os.path.join(all_paths_dic['diabetic_ret_test_path'],'negative'),normal_test)

  # calculating size of each sample:
  glaucoma_train_pos = len(os.listdir(os.path.join(os.getcwd(),os.path.join(all_paths_dic['glaucoma_train_path'],'positive'))))
  glaucoma_train_neg = len(os.listdir(os.path.join(os.getcwd(),os.path.join(all_paths_dic['glaucoma_train_path'],'negative'))))
  glaucoma_test_pos = len(os.listdir(os.path.join(os.getcwd(),os.path.join(all_paths_dic['glaucoma_test_path'],'positive'))))
  glaucoma_test_neg = len(os.listdir(os.path.join(os.getcwd(),os.path.join(all_paths_dic['glaucoma_test_path'],'negative'))))

  diabetic_ret_train_pos = len(os.listdir(os.path.join(os.getcwd(),os.path.join(all_paths_dic['diabetic_ret_train_path'],'positive'))))
  diabetic_ret_train_neg = len(os.listdir(os.path.join(os.getcwd(),os.path.join(all_paths_dic['diabetic_ret_train_path'],'negative'))))
  diabetic_ret_test_pos = len(os.listdir(os.path.join(os.getcwd(),os.path.join(all_paths_dic['diabetic_ret_test_path'],'positive'))))
  diabetic_ret_test_neg = len(os.listdir(os.path.join(os.getcwd(),os.path.join(all_paths_dic['diabetic_ret_test_path'],'negative'))))

  # Printing results:
  print("\n\nGlaucoma Retinopathy Summary:")
  print("\nGlaucoma positive cases in train dataset: ",glaucoma_train_pos)
  print("Glaucoma positive cases in test dataset: ",glaucoma_test_pos)
  print("Glaucoma negative cases in train dataset: ",glaucoma_train_neg)
  print("Glaucoma negative cases in test dataset: ",glaucoma_test_neg)

  print("\n\nDiabetic Retinopathy Summary:")
  print("\nDiabetic retinopathy positive cases in train dataset: ",diabetic_ret_train_pos)
  print("Diabetic retinopathy positive cases in test dataset: ",diabetic_ret_test_pos)
  print("Diabetic retinopathy negative cases in train dataset: ",diabetic_ret_train_neg)
  print("Diabetic retinopathy negative cases in test dataset: ",diabetic_ret_test_neg)



def get_data_summary(path_dic):
  # Function to get data summary 
  # Args:
      # path: path to train.csv summary file
  # returns: 
      # summary: pd.DataFrame() describing the instances of interest
  path = path_dic['summary_file_path']
  summary_path = get_data(path)
  df = pd.read_csv(summary_path,usecols=['filename','diabetic retinopathy','glaucoma','normal'])
  # useful idx
  glaucoma_and_diabetic_retinopathy_idx = (df['diabetic retinopathy'].values==1) & (df['glaucoma'].values==1)
  normal_idx  = df['normal'].values==1
  
  glaucoma_num = df['glaucoma'].values.sum()
  diabetic_retinopathy_num = df['diabetic retinopathy'].values.sum()
  glaucoma_and_diabetic_retinopathy_num = glaucoma_and_diabetic_retinopathy_idx.sum()
  background = df[normal_idx].shape[0]
  
  summary = pd.DataFrame([["glaucoma",'diabetic retinopathy','glaucoma & diabetic retinopathy','normal'],
                          [glaucoma_num,diabetic_retinopathy_num,glaucoma_and_diabetic_retinopathy_num,background]]).T
  
  summary.columns = ["instance name","number of cases"]
  summary.set_index("instance name",inplace=True)
  
  
  
  plt.figure(figsize=(15, 10))
  # Plotting
  # Defining the values for x-axis, y-axis
  # and from which datafarme the values are to be picked
  plots = sns.barplot(x=summary.index, y="number of cases", data=summary)

  # Iterrating over the bars one-by-one
  for bar in plots.patches:
      plots.annotate(format(bar.get_height(), '.2f'), 
                      (bar.get_x() + bar.get_width() / 2, 
                      bar.get_height()), ha='center', va='center',
                      size=15, xytext=(0, 8),
                      textcoords='offset points')

  # Setting the label for x-axis
  plt.xlabel("Pathologies", size=14)

  # Setting the label for y-axis
  plt.ylabel("Number of Cases", size=14)

  # Setting the title for the graph
  plt.title("Data Summary")
  
  return [summary,plt,df]




# Method to plot sample of data from training set
def plot_sampleData(data_path):
  # Method to plot samples of retinal images
  # Args:
    # data_path -> str representing path to the directory containing the images
  # Retuns:
    # --- plots samples
  nrows = 1
  ncols = 2
  pic_index = 0  
  fig = plt.gcf()
  fig.set_size_inches(ncols*6, nrows*4)
  sample_images = []
  directories = os.listdir(data_path)
  for i,dir in enumerate(directories):
    img_lst = os.listdir(os.path.join(data_path,dir))
    image_file = sample( img_lst,1)[0]
    image_path = os.path.join(data_path,dir,image_file)
    sp = plt.subplot(nrows, ncols, i + 1)
    sp.axis('Off') # Don't show axes (or gridlines)
    img = mpimg.imread(image_path)
    plt.imshow(img)
    plt.title(directories[i])
  plt.show()
  print("image size is:" + str(img.shape[0]) + "x" + str(img.shape[1]) + " pixels")



##################################################
#
#                       MODEL
#
##################################################

# Building Model 
def get_model(params,print_summary=False):
  # Mathod to build model
  # Args:
    # params -> dict holding model params
    # print_summary -> bool (default is False). if true, prints model summary design
  # Returns:
    #  Model object from tensorflow.keras.models

  # get params:
  number_of_fixed_layers = params['number_of_fixed_layers']
  l2_penalty = params['l2_penalty']
  pre_trained_model   = InceptionV3(weights='imagenet',include_top=True)
  # Freezing all layers until layer conv2D_181
  for layer in pre_trained_model.layers[:number_of_fixed_layers]: 
    layer.trainable=False
  last_layer  = pre_trained_model.get_layer('avg_pool')
  last_output = last_layer.output
  x = layers.Dense(1024,activation='relu')(last_output)
  x = layers.Dense(256,activation='relu')(x)
  x = layers.Dense(2,activation='softmax',kernel_regularizer=tf.keras.regularizers.l2(l2=l2_penalty))(x)
  model=Model(pre_trained_model.input,x)
  return model

def get_callbacks(checkpoint_dir,logs_dir):
  # Method to get callbacks
  # Args:
    # checkpoint_dir -> str representing checkpoint dir
    # logs_dir -> str representing log dir
  # Returns:
    # list of callbacks
  
  # Tensorboard callbacks
  tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logs_dir)

  # checkpoint callback
  model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath                = checkpoint_dir,
    save_freq               = 'epoch',
    save_weights_only       = True,
    monitor                 = 'val_accuracy',
    save_best_only          = True)
  return [tensorboard_callback , model_checkpoint_callback]


##################################################
#
#              PRE PROCESSING
#
##################################################
def get_generators(params,train_path,test_path,FinalTest_path):
  # method for pre-prcessing and providing true ground y_labels
  # Args:
    # params -> dict describing pre-processing parameters
  # returns:
    # generators -> dictionary of train and test generators as well as true y_label
  
  #get params:
  rotation_range = params['rotation_range']
  zca_whitening = params['zca_whitening']
  zca_epsilon = params['zca_epsilon']
  width_shift_range = params['width_shift_range']
  height_shift_range = params['height_shift_range']
  shear_range = params['shear_range']
  zoom_range = params['zoom_range']
  horizontal_flip = params['horizontal_flip']
  fill_mode = params['fill_mode']
  brightness_range = params['brightness_range']
  batch_size = params['batch_size']
  target_size = params['target_size']
 
  train_datagen=ImageDataGenerator(preprocessing_function = preprocess_input,
                                  rotation_range          = rotation_range,
                                  zca_whitening           = zca_whitening, 
                                  zca_epsilon             = zca_epsilon,
                                  width_shift_range       =width_shift_range,
                                  height_shift_range      =height_shift_range,
                                  shear_range             =shear_range,
                                  zoom_range              =zoom_range,
                                  horizontal_flip         =horizontal_flip,
                                  fill_mode               =fill_mode,
                                  brightness_range        = brightness_range)

  test_datagen  = ImageDataGenerator(preprocessing_function = preprocess_input)
  TEST_DATA_GEN = ImageDataGenerator(preprocessing_function = preprocess_input)

  generator_train      = train_datagen.flow_from_directory(train_path,target_size = target_size,batch_size=batch_size)
  generator_test       = test_datagen.flow_from_directory(test_path,target_size = target_size,batch_size=batch_size)
  label_generator      = test_datagen.flow_from_directory(test_path,target_size = target_size,batch_size=batch_size,shuffle=False)
  TEST_GEN_unlabeled   = TEST_DATA_GEN.flow_from_directory(FinalTest_path
                                                           ,target_size = target_size,batch_size=batch_size,shuffle=False) 

  generator = {'generator_train': generator_train,'generator_test':generator_test,'label_generator':label_generator, 'TEST_GEN_unlabeled':TEST_GEN_unlabeled}
  return generator


##################################################
#
#              Grad CAM
#
##################################################

def img_is_color(img):

    if len(img.shape) == 3:
        # Check the color channels to see if they're all the same.
        c1, c2, c3 = img[:, : , 0], img[:, :, 1], img[:, :, 2]
        if (c1 == c2).all() and (c2 == c3).all():
            return True

    return False
def show_image_list(list_images, list_titles=None, list_cmaps=None, grid=True, num_cols=2, figsize=(15, 7), title_fontsize=30):
    '''
    Shows a grid of images, where each image is a Numpy array. The images can be either
    RGB or grayscale.

    Parameters:
    ----------
    images: list
        List of the images to be displayed.
    list_titles: list or None
        Optional list of titles to be shown for each image.
    list_cmaps: list or None
        Optional list of cmap values for each image. If None, then cmap will be
        automatically inferred.
    grid: boolean
        If True, show a grid over each image
    num_cols: int
        Number of columns to show.
    figsize: tuple of width, height
        Value to be passed to pyplot.figure()
    title_fontsize: int
        Value to be passed to set_title().
    '''

    assert isinstance(list_images, list)
    assert len(list_images) > 0
    assert isinstance(list_images[0], np.ndarray)

    if list_titles is not None:
        assert isinstance(list_titles, list)
        assert len(list_images) == len(list_titles), '%d imgs != %d titles' % (len(list_images), len(list_titles))

    if list_cmaps is not None:
        assert isinstance(list_cmaps, list)
        assert len(list_images) == len(list_cmaps), '%d imgs != %d cmaps' % (len(list_images), len(list_cmaps))

    num_images  = len(list_images)
    num_cols    = min(num_images, num_cols)
    num_rows    = int(num_images / num_cols) + (1 if num_images % num_cols != 0 else 0)

    # Create a grid of subplots.
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    
    # Create list of axes for easy iteration.
    list_axes = list(axes.flat)

    for i in range(num_images):

        img    = list_images[i]
        title  = list_titles[i] if list_titles is not None else 'Image %d' % (i)
        cmap   = list_cmaps[i] if list_cmaps is not None else (None if img_is_color(img) else 'gray')
        
        list_axes[i].imshow(img, cmap=cmap)
        list_axes[i].set_title(title, fontsize=title_fontsize) 
        list_axes[i].grid(grid)

    for i in range(num_images, len(list_axes)):
        list_axes[i].set_visible(False)

    fig.tight_layout()
    _ = plt.show()

###############################################################################
#    gradCam
###############################################################################

    
def gradCAM(model=None,test_path=None,image_num = 0, intensity=0.5, res=250):
    
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    generator      = datagen.flow_from_directory(test_path,target_size = (299,299),batch_size=250,shuffle=False)
    images,_ = next(generator)
    images,_ = next(generator)

    # Path to original image
    path_to_image = test_path + "/"+generator.filenames[image_num]
    
    x = images[image_num]
    x=np.expand_dims(x, axis=0)
    preds = model.predict(x)

    with tf.GradientTape() as tape:
        last_conv_layer = model.get_layer('mixed10')
        iterate = tf.keras.models.Model([model.inputs], [model.output, last_conv_layer.output])
        model_out, last_conv_layer = iterate(x)
        class_out = model_out[:, np.argmax(model_out[0])]
        grads = tape.gradient(class_out, last_conv_layer)
        pooled_grads = K.mean(grads, axis=(0, 1, 2))

    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = heatmap.reshape((8,8))
    
    orig = cv2.imread(path_to_image)
    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)


    heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)

    img = heatmap * intensity + orig
    show_image_list([orig.astype('uint8'),img.astype('uint8')],list_titles=["Original","gradCam Applied"])
  

##################################################
#
#              Semi-Supervised Split
#
##################################################

def semi_supervised_split(clf,src_generator,model='glaucoma'):
  # Method to copy files based on predicted earlier models
  # Args:
  #   clf: classifier
  #   src_generator: Source generator flowing data from test directory
  #   model : str can be either glaucoma or retinopathy

  y_proba = clf.predict(src_generator)
  y_pred  = y_proba.argmax(axis=1)

  src_path_base     = os.path.join(all_paths_dic['test_train_path'],'test')
  dst_path_base ='/content/glaucoma_data/train' if model=='glaucoma' else '/content/diabetic_ret_data/train'

  file_list = os.listdir(src_path_base)

  for idx,label in enumerate(y_pred):
    file_name = file_list[idx]
    src = os.path.join(src_path_base,file_name)
    if label==1:
      dst = os.path.join(dst_path_base,'positive')
    else:
      dst = os.path.join(dst_path_base,'negative')
    shutil.copy(src,dst)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger('tensorflow').disabled = True
import setproctitle
SERVER_NAME = 'ultrafast'
import time
import sys
import shutil
sys.path.append('lib/TASK_2_UC1/')
from models import *
from util import otsu_thresholding
from functions import *
sys.path.append('lib/')
from mlta import *
import math
from keras.callbacks import Callback
import sklearn.metrics
#
EXPERIMENT_TYPE=sys.argv[2]
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
keras.backend.set_session(tf.Session(config=config))
#
verbose=True 
# ORIGINAL DATA PATHS
#cam16 = hd.File('/home/mara/adversarialMICCAI/data/ultrafast/cam16_500/patches.hdf5',  'r', libver='latest', swmr=True)
#all500 = hd.File('/home/mara/adversarialMICCAI/data/ultrafast/all500/patches.hdf5',  'r', libver='latest', swmr=True)
#extra17 = hd.File('/home/mara/adversarialMICCAI/data/ultrafast/extra17/patches.hdf5',  'r', libver='latest', swmr=True)
#tumor_extra17=hd.File('/home/mara/adversarialMICCAI/data/ultrafast/1129-1155/patches.hdf5', 'r', libver='latest', swmr=True)
#test2 = hd.File('/mnt/nas2/results/IntermediateResults/Camelyon/ultrafast/test_data2/patches.hdf5', 'r', libver='latest', swmr=True)
#pannuke= hd.File('/mnt/nas2/results/IntermediateResults/Camelyon/pannuke/patches_fix.hdf5', 'r', libver='latest', swmr=True)
#data={'cam16':cam16,'all500':all500,'extra17':extra17, 'tumor_extra17':tumor_extra17, 'test_data2': test2, 'pannuke':pannuke}
global data
data = hd.File('data/demo_file.h5py', 'r') # Replace here with your data 
CONFIG_FILE = 'doc/config.cfg'
COLOR = True
global new_folder
new_folder = getFolderName()
folder_name= EXPERIMENT_TYPE
new_folder = 'results/'+folder_name
BATCH_SIZE = 32
f=open(new_folder+"/seed.txt","w")
seed=int(sys.argv[1])
print seed
f.write(str(seed))
f.close()
# SET PROCESS TITLE
setproctitle.setproctitle('UC1_{}'.format(EXPERIMENT_TYPE))
# SET SEED
np.random.seed(seed)
tf.set_random_seed(seed)
# DATA SPLIT CSVs
train_csv=open('doc/demo_train_shuffle.csv', 'r') 
val_csv=open('doc/demo_val_shuffle.csv', 'r')
test_csv=open('doc/demo_test_shuffle.csv', 'r')
train_list=train_csv.readlines()
val_list=val_csv.readlines()
test_list=test_csv.readlines()
test2_csv = open('doc/demo_test2_shuffle.csv', 'r')
test2_list=test2_csv.readlines()
test2_csv.close()
train_csv.close()
val_csv.close()
test_csv.close()
data_csv=open('./doc/demo_data_shuffle.csv')
data_list=data_csv.readlines()
data_csv.close()
# STAIN NORMALIZATION
def get_normalizer(patch, save_folder=''):
    normalizer = ReinhardNormalizer()
    normalizer.fit(patch)
    np.save('{}/normalizer'.format(save_folder),normalizer)
    np.save('{}/normalizing_patch'.format(save_folder), patch)
    print('Normalisers saved to disk.')
    return normalizer
def normalize_patch(patch, normalizer):
    return np.float64(normalizer.transform(np.uint8(patch)))
# LOAD DATA NORMALIZER
global normalizer
db_name, entry_path, patch_no = get_keys(data_list[0])
normalization_reference_patch = data[db_name][entry_path][str(patch_no)][:]
normalizer = get_normalizer(normalization_reference_patch, save_folder=new_folder)

"""
Batch generators:
They load a patch list: a list of file names and paths.
They use the list to create a batch of 32 samples.
"""
# BATCH GENERATORS
def get_batch_data(patch_list, batch_size=32):
    num_samples=len(patch_list)
    while True:
        offset = 0
        for offset in range(0,num_samples, batch_size):
            batch_x = []
            batch_y = []
            batch_cm = []
            batch_samples=patch_list[offset:offset+batch_size]
            for line in batch_samples:
                db_name, entry_path, patch_no = get_keys(line)
                patch=data[db_name][entry_path][patch_no][:]
                patch=normalize_patch(patch, normalizer)
                patch=keras.applications.inception_v3.preprocess_input(patch) 
                label = get_class(line, entry_path) 
                batch_x.append(patch)
                batch_y.append(label)
            batch_x = np.asarray(batch_x, dtype=np.float32)
            yield np.asarray(batch_x, dtype=np.float32), np.asarray(batch_y, dtype=np.float32)       

def get_test_batch(patch_list, batch_size=32):
    num_samples=len(patch_list)
    while True:
        for offset in range(0,num_samples, batch_size):
            batch_x = []
            batch_y = []
            batch_cm = []
            batch_samples=patch_list[offset:offset+batch_size]
            for line in batch_samples:
                db_name, entry_path, patch_no = get_keys(line)
                #import pdb; pdb.set_trace()
                patch=data[db_name][entry_path][patch_no][:]
                patch=normalize_patch(patch, normalizer)
                patch=keras.applications.inception_v3.preprocess_input(patch)
                label = get_test_label(entry_path)
                batch_x.append(patch)
                batch_y.append(label)
            batch_x = np.asarray(batch_x, dtype=np.float32)
            yield np.asarray(batch_x, dtype=np.float32), np.asarray(batch_y, dtype=np.float32)

"""
Building baseline model
"""
base_model = keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet', input_shape=(224,224,3))
layers_list=['conv2d_92', 'conv2d_93', 'conv2d_88', 'conv2d_89', 'conv2d_86']
for i in range(len(base_model.layers[:])):
    layer=base_model.layers[i]
    if layer.name in layers_list:
        print layer.name
        layer.trainable=True
    else:
        layer.trainable = False
feature_output=base_model.layers[-1].output
feature_output = keras.layers.GlobalAveragePooling2D()(feature_output)
feature_output = Dense(2048, activation='relu', name='finetuned_features1',kernel_regularizer=keras.regularizers.l2(0.01))(feature_output) 
feature_output = keras.layers.Dropout(0.8, noise_shape=None, seed=None)(feature_output)
feature_output = Dense(512, activation='relu', name='finetuned_features2',kernel_regularizer=keras.regularizers.l2(0.01))(feature_output)
feature_output = keras.layers.Dropout(0.8, noise_shape=None, seed=None)(feature_output)
feature_output = Dense(256, activation='relu', name='finetuned_features3',kernel_regularizer=keras.regularizers.l2(0.01))(feature_output)
feature_output = keras.layers.Dropout(0.8, noise_shape=None, seed=None)(feature_output)
finetuning = Dense(1,name='predictions')(feature_output)
model = Model(input=base_model.input, output=finetuning)
#
def compile_model(model, opt):
    model.compile(optimizer=opt,
                  loss=classifier_loss,
                  metrics=[my_acc_f, my_accuracy,tp_count, fp_count, fn_count, tn_count]
                 )
# Callbacks
logdir='{}/tb_log'.format(new_folder)
checkpoint_dir = '{}'.format(new_folder) 
callbacks = [LR_scheduling(new_folder=new_folder), 
             keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10),
            keras.callbacks.ModelCheckpoint('{}/best_model.h5'.format(checkpoint_dir), monitor='val_loss', mode='min', save_best_only=True, verbose=1)]

""" Logging files"""
f=open("{}/{}_log.txt".format(new_folder, EXPERIMENT_TYPE), 'w')
print("{}/{}_log.txt".format(new_folder, EXPERIMENT_TYPE))
f.write('Trainable layers: ')
for layer in model.layers:
    if layer.trainable==True:
        f.write('{}\n'.format(layer.name))
f.close()
global lr_monitor
lr_monitor=open('{}/lr_monitor.log'.format(new_folder), 'w') #if hvd.rank()==0 else None
print("Opened monitoring file for learning rate: {}/lr_monitor.log".format(new_folder))
""" """

initial_lr = 1e-4
opt = keras.optimizers.SGD(lr=initial_lr, momentum=0.9, nesterov=False)
compile_model(model,opt)

""" Init Generators """
train_generator=get_batch_data(data_list, batch_size=BATCH_SIZE)
val_generator=get_test_batch(val_list, batch_size=BATCH_SIZE)
test_generator=get_test_batch(test_list, batch_size=BATCH_SIZE)
""" """
avg_auc = 0
T_B=len(test_list)//BATCH_SIZE
ys=np.zeros(len(test_list))
preds=np.zeros((len(test_list),1))
#
for i in range(T_B):
    x,y=test_generator.next()
    ys[i*BATCH_SIZE:(i)*BATCH_SIZE+len(y)] = y
    preds[i*BATCH_SIZE:(i)*BATCH_SIZE+len(y)] = model.predict(x)
preds=tf.sigmoid(preds).eval(session=tf.Session())
try:
    auc=sklearn.metrics.roc_auc_score(ys,preds)
except:
    print("auc score not available in demo mode")
    auc=0.
    
print 'Before training auc: ', auc
rpreds=tf.round(preds).eval(session=tf.Session())
before_training_score=sklearn.metrics.accuracy_score(ys, rpreds)
f=open("{}/{}_log.txt".format(new_folder, EXPERIMENT_TYPE), 'a')
f.write('Before training auc: {}\n'.format(auc))
f.write('Before training loss, acc: {}'.format(before_training_score))
f.close()

# START TRAINING
starting_time = time.time()

history = model.fit_generator(train_generator,
                    steps_per_epoch= len(data_list)// BATCH_SIZE,# // hvd.size(),
                    callbacks=callbacks,
                    epochs=100,
                    verbose=verbose,
                    workers=1,
                    use_multiprocessing=False,
                    validation_data=val_generator,
                    validation_steps= len(val_list)// BATCH_SIZE # // hvd.size()
                             )
end_time = time.time()
lr_monitor.close()

total_training_time = end_time - starting_time
#
avg_auc = 0
T_B=len(test_list)//BATCH_SIZE
ys=np.zeros(len(test_list))
preds=np.zeros((len(test_list),1))
for i in range(T_B):
    x,y=test_generator.next()
    ys[i*BATCH_SIZE:(i)*BATCH_SIZE+len(y)] = y
    preds[i*BATCH_SIZE:(i)*BATCH_SIZE+len(y)] = model.predict(x)
preds = tf.sigmoid(preds).eval(session=tf.Session())
auc = sklearn.metrics.roc_auc_score(ys,preds)
accuracy = sklearn.metrics.accuracy_score(ys, tf.round(preds).eval(session=tf.Session()))
#
if verbose:
    print ('Before training test score: ', before_training_score)
    print('Test accuracy:', accuracy)
    f=open("{}/{}_log.txt".format(new_folder, EXPERIMENT_TYPE), 'a')
    f.write('Post training loss, acc: {}\n'.format(accuracy))
    f.close()
    print 'Post training auc: ', auc
    f=open("{}/{}_log.txt".format(new_folder, EXPERIMENT_TYPE), 'a')
    f.write('Post training auc: {}\n'.format(auc))
#
print history.history.keys()
for k in history.history.keys():
     f.write('{}\n'.format(history.history[k]))
f.write('Time elapsed for model training: {}\n'.format(total_training_time))
f.close()
np.save('{}/training_log'.format(new_folder), history.history)
print "++++++ BASELINE: OK WE'RE DONE OVER HERE ++++++ FOLDER: {} ++++++ GOOD JOB. ".format(new_folder)
exit(0)

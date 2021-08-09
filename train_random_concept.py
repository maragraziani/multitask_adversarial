"""
RUN SEQUENTIAL
python hvd_train_unc.py SEED EXPERIMENT_TYPE 
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger('tensorflow').disabled = True
from keras import *
import setproctitle
SERVER_NAME = 'ultrafast'
import time
import sys
import shutil
sys.path.append('lib/TASK_2_UC1/')
from models import *
from util import otsu_thresholding
from extract_xml import *
from functions import *                   
sys.path.append('lib/')
from mlta import *
import math
import keras.callbacks as callbacks
from keras.callbacks import Callback
import tensorflow as tf

EXPERIMENT_TYPE=sys.argv[2]
CONCEPT=sys.argv[3]
c = CONCEPT.split(',')
c_list=[]
for c_ in c:
    c_list.append(c_.strip('[').strip(']'))
print 'concept', CONCEPT, c, c_list
CONCEPT=c_list
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(0)# str(hvd.local_rank())
keras.backend.set_session(tf.Session(config=config))
verbose=1 

cam16 = hd.File('/home/mara/adversarialMICCAI/data/ultrafast/cam16_500/patches.h5py',  'r', libver='latest', swmr=True)
all500 = hd.File('/home/mara/adversarialMICCAI/data/ultrafast/all500/patches.h5py',  'r', libver='latest', swmr=True)
extra17 = hd.File('/home/mara/adversarialMICCAI/data/ultrafast/extra17/patches.h5py',  'r', libver='latest', swmr=True)
tumor_extra17=hd.File('/home/mara/adversarialMICCAI/data/ultrafast/1129-1155/patches.h5py', 'r', libver='latest', swmr=True)
test2 = hd.File('/home/mara/adversarialMICCAI/data/ultrafast/test_data2/patches.h5py', 'r', libver='latest', swmr=True)
pannuke= hd.File('/home/mara/adversarialMICCAI/data/ultrafast/pannuke/patches_fix.h5py', 'r', libver='latest', swmr=True)
global data
data={'cam16':cam16,'all500':all500,'extra17':extra17, 'tumor_extra17':tumor_extra17, 'test_data2': test2, 'pannuke':pannuke}
global concept_db
concept_db = hd.File('data/normalized_cmeasures/concept_values_def.h5py','r', libver='latest', swmr=True)

CONFIG_FILE = 'doc/config.cfg'
COLOR = True
global new_folder
new_folder = getFolderName()
folder_name=EXPERIMENT_TYPE
new_folder = 'results/'+folder_name #new_folder
global error_log
error_log=open(new_folder+'/ERR.log', 'w')
BATCH_SIZE = 32

# SAVE FOLD
f=open(new_folder+"/seed.txt","w")
seed=int(sys.argv[1])
if verbose:  print(seed)
f.write(str(seed))
f.close()
f = open('{}/val_by_epoch.txt'.format(new_folder), 'w')
ft = open('{}/train_by_epoch.txt'.format(new_folder), 'w')
f.close()
ft.close()
    
# SET PROCESS TITLE
setproctitle.setproctitle('{}'.format(EXPERIMENT_TYPE))

# SET SEED
np.random.seed(seed)
tf.set_random_seed(seed)
global SEED
SEED=seed

# DATA SPLIT CSVs 
train_csv=open('/mnt/nas2/results/IntermediateResults/Camelyon/train_shuffle.csv', 'r') # How is the encoding of .csv files ?
val_csv=open('/mnt/nas2/results/IntermediateResults/Camelyon/val_shuffle.csv', 'r')
test_csv=open('/mnt/nas2/results/IntermediateResults/Camelyon/test_shuffle.csv', 'r')
train_list=train_csv.readlines()
val_list=val_csv.readlines()
test_list=test_csv.readlines()
test2_csv = open('/mnt/nas2/results/IntermediateResults/Camelyon/test2_shuffle.csv', 'r')
test2_list=test2_csv.readlines()
test2_csv.close()
train_csv.close()
val_csv.close()
test_csv.close()
data_csv=open('./doc/data_shuffle.csv', 'r')
data_list=data_csv.readlines()
data_csv.close()
flog=open("{}/{}_log.txt".format(new_folder, EXPERIMENT_TYPE), 'w')
flog.write('/mnt/nas2/results/IntermediateResults/Camelyon/train_shuffle.csv')
flog.write('/mnt/nas2/results/IntermediateResults/Camelyon/val_shuffle.csv')
flog.write('/mnt/nas2/results/IntermediateResults/Camelyon/test_shuffle.csv')
flog.write('/mnt/nas2/results/IntermediateResults/Camelyon/test2_shuffle.csv')
flog.write('./doc/pannuke_data_shuffle.csv')
flog.close()

# STAIN NORMALIZATION
def get_normalizer(patch, save_folder=''):
    normalizer = ReinhardNormalizer()
    normalizer.fit(patch)
    np.save('{}/normalizer'.format(save_folder),normalizer)
    np.save('{}/normalizing_patch'.format(save_folder), patch)
    #print('Normalisers saved to disk.')
    return normalizer
def normalize_patch(patch, normalizer):
    return np.float64(normalizer.transform(np.uint8(patch)))

# LOAD DATA NORMALIZER
global normalizer
db_name, entry_path, patch_no = get_keys(data_list[0])
normalization_reference_patch = data[db_name][entry_path][patch_no]
normalizer = get_normalizer(normalization_reference_patch, save_folder=new_folder)

"""
Batch generators: 
They load a patch list: a list of file names and paths. 
They use the list to create a batch of 32 samples. 
"""

# Retrieve Concept Measures
def get_concept_measure(db_name, entry_path, patch_no, measure_type=''):
    if measure_type=='domain':
        return get_domain(db_name, entry_path)
    if measure_type=='random':
        #return np.random.rand()
        #np.random.default_rng(SEED)
        return np.random.standard_normal()
    path=db_name+'/'+entry_path+'/'+str(patch_no)+'/'+measure_type.strip(' ')
    try:
        cm=concept_db[path][0]
        return cm
    except:
        print("[ERR]: {}, {}, {}, {} with path {}".format(db_name, entry_path, patch_no, measure_type, path))
        return 1.
    
# BATCH GENERATORS
class DataGenerator(keras.utils.Sequence):
    def __init__(self, patch_list, concept=CONCEPT, batch_size=32, shuffle=True, data_type=0):
        self.batch_size=batch_size
        self.patch_list=patch_list
        self.shuffle=shuffle
        self.concept = concept
        self.data_type=data_type
        #print 'data type:', data_type
        self.on_epoch_end()
        
    def __len__(self):
        return int(np.floor(len(self.patch_list)/self.batch_size))
    def __getitem__(self, index):
        indexes=self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        patch_list_temp=[self.patch_list[k] for k in indexes]
        self.patch_list_temp=patch_list_temp
        return self.__data_generation(self), None
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.patch_list))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    def __data_generation(self, patch_list_temp):
        patch_list_temp=self.patch_list_temp
        batch_x=np.zeros((len(patch_list_temp), 224,224,3))
        batch_y=np.zeros(len(patch_list_temp))
        i=0
        for line in patch_list_temp:
            db_name, entry_path, patch_no = get_keys(line)
            patch=data[db_name][entry_path][patch_no]
            patch=normalize_patch(patch, normalizer)
            patch=keras.applications.inception_v3.preprocess_input(patch) 
            label = get_class(line, entry_path) 
            if self.data_type!=0:
                label=get_test_label(entry_path)
            batch_x[i]=patch
            batch_y[i]=label
            i+=1
        generator_output=[batch_x, batch_y]
        for c in self.concept:
            batch_concept_values=np.zeros(len(patch_list_temp))
            i=0
            for line in patch_list_temp:
                db_name, entry_path, patch_no = get_keys(line)
                batch_concept_values[i]=get_concept_measure(db_name, entry_path, patch_no, measure_type=c)
                i+=1
            if c=='domain':
                    batch_concept_values=keras.utils.to_categorical(batch_concept_values, num_classes=7)
            generator_output.append(batch_concept_values)
        return generator_output

"""
Credits for the Gradient Reversal Layer
https://github.com/michetonu/gradient_reversal_keras_tf/blob/master/flipGradientTF.py
"""
def reverse_gradient(X, hp_lambda):
    '''Flips the sign of the incoming gradient during training.'''
    hp_lambda = hp_lambda
    try:
        reverse_gradient.num_calls += 1
    except AttributeError:
        reverse_gradient.num_calls = 1
    grad_name = "GradientReversal%d" % reverse_gradient.num_calls
    @tf.RegisterGradient(grad_name)
    def _flip_gradients(op, grad):
        grad = tf.negative(grad)
        #grad = tf.Print(grad, [grad], 'grad')
        final_val = grad * hp_lambda 
        #final_val =tf.Print(final_val, [final_val], 'final_val')
        return [final_val]
    g = keras.backend.get_session().graph
    with g.gradient_override_map({'Identity': grad_name}):
        y = tf.identity(X)
    return y

class Hp_lambda():
    def __init__(self,in_val):
        self.value=in_val
    def update(self,new_val):
        self.value=new_val
    def get_hyperparameter_lambda(self):
        #val = tf.Print(self.value,[self.value],'hplambda: ')
        return tf.Variable(self.value, name='hp_lambda')
    #return lmb
class GradientReversal(Layer):
    '''Flip the sign of gradient during training.'''
    def __init__(self, hp_lambda, **kwargs):
        super(GradientReversal, self).__init__(**kwargs)
        self.supports_masking = False
        self.hp_lambda = Hp_lambda(hp_lambda)
        #self.hp_lambda = tf.Variable(hp_lambda, name='hp_lambda')
        #self.hp_lambda = tf.Variable(hp_lambda, name='hp_lambda')
    def build(self, input_shape):
        self.trainable_weights = []
        return
    def call(self, x, mask=None):
        #tf.Print(self.hp_lambda, [self.hp_lambda],'self.hp_lambda: ')
        #with tf.Session() as sess:  print(self.hp_lambda.eval()) 
        lmb=self.hp_lambda.get_hyperparameter_lambda()
        return reverse_gradient(x, lmb)

    def get_output_shape_for(self, input_shape):
        return input_shape
    def get_config(self):
        config = {"name": self.__class__.__name__,
                  'hp_lambda': keras.backend.get_value(self.hp_lambda)}
        base_config = super(GradientReversal, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

"""         
Building guidable model 
"""
def get_baseline_model(hp_lambda=0., c_list=[]):
    base_model = keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet', input_shape=(224,224,3))
    layers_list=['conv2d_92', 'conv2d_93', 'conv2d_88', 'conv2d_89', 'conv2d_86']
    #layers_list=[]
    for i in range(len(base_model.layers[:])):
        layer=base_model.layers[i]
        if layer.name in layers_list:
            print layer.name
            layer.trainable=True
        else:
            layer.trainable = False
    feature_output=base_model.layers[-1].output
    gap_layer_output = keras.layers.GlobalAveragePooling2D()(feature_output)
    feature_output = Dense(2048, activation='relu', name='finetuned_features1',kernel_regularizer=keras.regularizers.l2(0.01))(gap_layer_output) 
    feature_output = keras.layers.Dropout(0.8, noise_shape=None, seed=None)(feature_output)
    feature_output = Dense(512, activation='relu', name='finetuned_features2',kernel_regularizer=keras.regularizers.l2(0.01))(feature_output)
    feature_output = keras.layers.Dropout(0.8, noise_shape=None, seed=None)(feature_output)
    feature_output = Dense(256, activation='relu', name='finetuned_features3',kernel_regularizer=keras.regularizers.l2(0.01))(feature_output)
    feature_output = keras.layers.Dropout(0.8, noise_shape=None, seed=None)(feature_output)
    #grl_layer=GradientReversal(hp_lambda=hp_lambda)
    #feature_output_grl = grl_layer(feature_output)
    #domain_adversarial = keras.layers.Dense(7, activation = keras.layers.Activation('softmax'), name='domain_adversarial')(feature_output_grl)
    finetuning = Dense(1,name='predictions')(feature_output)
    ## here you need to check how many other concepts you have apart from domain adversarial
    # then you add one layer per each. 
    output_nodes=[finetuning]#, domain_adversarial]
    for c in c_list:
        if c!='domain':
            concept_layer=  keras.layers.Dense(1, activation = keras.layers.Activation('linear'), name='extra_{}'.format(c.strip(' ')))(feature_output)
            output_nodes.append(concept_layer)
    model = Model(input=base_model.input, output=output_nodes)
    #model.grl_layer=grl_layer
    return model

"""
CALLBACKS
"""
def compute_mse(labels, predictions):
    errors = labels - predictions
    sum_squared_errors = np.sum(np.asarray([pow(errors[i],2) for i in range(len(errors))]))
    mse = sum_squared_errors / len(labels)
    return mse

def evaluate(pred_, save_file=None):
    y_true = pred_[:,0]
    #domain_true = pred_[:,1:8]
    true_extra_concepts={}
    if len(c_list)>0:
        for i in range(1, len(c_list)):
            true_extra_concepts[i]=pred_[:,8+i]
    #print(i)
    y_pred = pred_[:,8+i]
    val_acc = my_accuracy_np(y_true, y_pred)
    #domain_pred = pred_[:, 8+i+1:8+i+1+7]
    last_index=8+i#+7
    pred_extra_concepts={}
    if len(c_list)>0:
        for i in range(1, len(c_list)):
            pred_extra_concepts[i]=pred_[:,last_index+i]
    #val_acc_d = accuracy_domain(domain_true, domain_pred)
    val_r2={}
    val_mse={}
    if len(c_list)>0:
        for i in range(1, len(c_list)):
            val_r2[i] = r_square_np(true_extra_concepts[i], pred_extra_concepts[i])
            val_mse[i] = compute_mse(true_extra_concepts[i], pred_extra_concepts[i])
    
    extra_string=''
    if len(c_list)>0:
        for i in range(1, len(c_list)):
            extra_string=extra_string+" {}: r2 {}, mse {}; ".format(i, val_r2[i], val_mse[i])
    #print("Acc: {}, Acc domain: {}\n".format(val_acc, val_acc_d)+extra_string)
    #save_file.write("Val acc: {}, acc_domain: {}\n".format(val_acc, val_acc_d)+extra_string)
    

class eval_model(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        f = open('{}/val_by_epoch.txt'.format(new_folder), 'a')
        ft = open('{}/train_by_epoch.txt'.format(new_folder), 'a')
        pred_ = self.model.predict_generator(val_generator2, steps = len(val_list)// BATCH_SIZE, workers=4, use_multiprocessing=False)#// hvd.size()) 
        #import pdb; pdb.set_trace()
        y_true = pred_[:,0]
        y_pred = pred_[:,2]
        #print 'y_true: ', y_true
        #print 'y_pred: ', y_pred
        val_acc = my_accuracy_np(y_true, y_pred)
        cm_true = pred_[:, 1]
        cm_pred = pred_[:,3]
        #print 'cm_true: ', cm_true
        #print 'cm_pred: ', cm_pred
        
        val_r2 = r_square_np(cm_true, cm_pred)
        val_mse = compute_mse(cm_true, cm_pred)
        print("Val acc: {}, r2: {}, MSE: {}\n".format(val_acc, val_r2, val_mse))
        report_val_acc.append(val_acc)
        report_val_r2.append(val_r2)
        report_val_mse.append(val_mse)
        f.write("Val acc: {}, r2: {}, mse: {}\n".format(val_acc, val_r2, val_mse))
        
        train_pred_ = self.model.predict_generator(train_generator2, steps=100, workers=1, use_multiprocessing=False)
        cm_true = train_pred_[:, 1]
        cm_pred = train_pred_[:,3]
        #print 'train_cmtrue: ', cm_true
        #print 'train_cmpred: ', cm_pred
        train_r2 = r_square_np(cm_true, cm_pred)
        train_mse = compute_mse(cm_true, cm_pred)
        print("Train r2: {}, MSE: {}".format(train_r2, train_mse))
        ft.write("Train acc: {}, r2: {}, mse: {}\n".format(val_acc, val_r2, val_mse))
      
logdir='{}/tb_log'.format(new_folder)
checkpoint_dir = '{}'.format(new_folder) 
callbacks = [LR_scheduling(new_folder=new_folder, loss=None, metrics=None), 
             eval_model(),
             keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, 
                                           patience=10),
            keras.callbacks.ModelCheckpoint('{}/best_model.h5'.format(checkpoint_dir), monitor='val_loss', mode='min', save_best_only=True, save_weights_only=True, verbose=1)
            ]
# END Callbacks
""" 
LOSS FUNCTIONS
"""
def keras_mse(y_true, y_pred):
    return tf.reduce_mean(tf.keras.losses.mean_squared_error(y_true, y_pred))

def bbce(y_true, y_pred):
    # we use zero weights to set the loss to zero for unlabeled data
    verbose=0
    zero= tf.constant(-1, dtype=tf.float32)
    where = tf.not_equal(y_true, zero)
    where = tf.reshape(where, [-1])
    indices=tf.where(where) #indices where the item of y_true is NOT -1
    indices = tf.reshape(indices, [-1])
    sliced_y_true = tf.nn.embedding_lookup(y_true, indices)
    sliced_y_pred = tf.nn.embedding_lookup(y_pred, indices)
    n1 = tf.shape(indices)[0] #number of train images in batch
    batch_size = tf.shape(y_true)[0]
    n2 = batch_size - n1 #number of test images in batch
    sliced_y_true = tf.reshape(sliced_y_true, [n1, -1])
    n1_ = tf.cast(n1, tf.float32)
    n2_ = tf.cast(n2, tf.float32)
    multiplier = (n1_+ n2_) / n1_
    zero_class = tf.constant(0, dtype=tf.float32)
    where_class_is_zero=tf.cast(tf.reduce_sum(tf.cast(tf.equal(sliced_y_true, zero_class), dtype=tf.float32)), dtype=tf.float32)
    if verbose:
        where_class_is_zero=tf.Print(where_class_is_zero,[where_class_is_zero],'where_class_is_zero: ')
    class_weight_zero = tf.cast(tf.divide(n1_, 2. * tf.cast(where_class_is_zero, dtype=tf.float32)+0.001), dtype=tf.float32)
    
    if verbose:
        class_weight_zero=tf.Print(class_weight_zero,[class_weight_zero],'class_weight_zero: ')
    one_class = tf.constant(1, dtype=tf.float32)
    where_class_is_one=tf.cast(tf.reduce_sum(tf.cast(tf.equal(sliced_y_true, one_class), dtype=tf.float32)), dtype=tf.float32)
    if verbose:
        where_class_is_one=tf.Print(where_class_is_one,[where_class_is_one],'where_class_is_one: ')
        n1_=tf.Print(n1_,[n1_],'n1_: ')
    class_weight_one = tf.cast(tf.divide(n1_, 2. * tf.cast(where_class_is_one,dtype=tf.float32)+0.001), dtype=tf.float32)
    class_weight_zero =  tf.constant(23477.0/(23477.0+123820.0), dtype=tf.float32)
    class_weight_one =  tf.constant(123820.0/(23477.0+123820.0), dtype=tf.float32)
    A = tf.ones(tf.shape(sliced_y_true), dtype=tf.float32) - sliced_y_true 
    A = tf.scalar_mul(class_weight_zero, A)
    B = tf.scalar_mul(class_weight_one, sliced_y_true) 
    class_weight_vector=A+B
    ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=sliced_y_true,logits=sliced_y_pred)
    ce = tf.multiply(class_weight_vector,ce)
    return tf.reduce_mean(ce)

from keras.initializers import Constant
global domain_weight
global main_task_weight

class CustomMultiLossLayer(Layer):
    def __init__(self, new_folder='', nb_outputs=2, **kwargs):
        self.nb_outputs = nb_outputs
        self.is_placeholder = True
        super(CustomMultiLossLayer, self).__init__(**kwargs)
        
    def build(self, input_shape=None):
        # initialise log_vars
        self.log_vars = []
        for i in range(self.nb_outputs):
            self.log_vars += [self.add_weight(name='log_var' + str(i), shape=(1,),
                                              initializer=Constant(0.), trainable=True)]
        super(CustomMultiLossLayer, self).build(input_shape)
    """
    def multi_loss(self, ys_true, ys_pred):
        assert len(ys_true) == self.nb_outputs and len(ys_pred) == self.nb_outputs
        loss = 0
        for y_true, y_pred, log_var in zip(ys_true, ys_pred, self.log_vars):
            precision = K.exp(-log_var[0])
            loss += K.sum(precision * (y_true - y_pred)**2. + log_var[0], -1)
        return K.mean(loss)
    """
    def multi_loss(self,  ys_true, ys_pred):
        assert len(ys_true) == self.nb_outputs and len(ys_pred) == self.nb_outputs
        loss = 0
        i=0
        for y_true, y_pred, log_var in zip(ys_true, ys_pred, self.log_vars):
            precision =keras.backend.exp(-log_var[0]) 
            if i==0:
                pred_loss = bbce(y_true, y_pred)
                term = main_task_weight*precision*pred_loss + main_task_weight*0.5 * log_var[0]  
                #term=tf.Print(keras.backend.mean(term), [keras.backend.mean(term)], 'mean bbce: ')
            #elif i==1:
                # I need to find a better way for this
            #    pred_loss = keras.losses.categorical_crossentropy(y_true, y_pred)
                #keras_mse(y_true, y_pred)
            #    term =  domain_weight * precision * pred_loss + domain_weight * log_var[0]
                #term=tf.Print(keras.backend.mean(term), [keras.backend.mean(term)], 'mean cce: ')
            else:
                pred_loss = keras_mse(y_true, y_pred)
                #pred_loss=tf.Print(pred_loss, [pred_loss], 'MSE: ')
                term = 0.5 * precision * pred_loss + 0.5 * log_var[0]
            loss+=term
            term = 0.
            i+=1
        return keras.backend.mean(loss)
    
    def call(self, inputs):
        ys_true = inputs[:self.nb_outputs]
        ys_pred = inputs[self.nb_outputs:]
        loss = self.multi_loss(ys_true, ys_pred)
        self.add_loss(loss, inputs=inputs)
        return keras.backend.concatenate(inputs, -1)

"""
EVALUATION FUNCTIONs
"""
def accuracy_domain(y_true,y_pred):
    #y_p_r=
    y_p_r = np.asarray([np.argmax(y_pred[i,:]) for i in range(len(y_pred[:,0]))])
    #y_p_r=np.round(y_pred)
    y_true = np.asarray([np.argmax(y_true[i,:]) for i in range(len(y_true[:,0]))])
    acc = np.equal(y_p_r, y_true)**1.
    acc = np.mean(np.float32(acc))
    return acc
def my_sigmoid(x):
    return 1 / (1 + np.exp(-x))
def my_accuracy_np(y_true, y_pred):
    sliced_y_pred = my_sigmoid(y_pred)
    y_pred_rounded = np.round(sliced_y_pred)
    acc = np.equal(y_pred_rounded, y_true)**1.
    acc = np.mean(np.float32(acc))
    return acc
def r_square_np(y_true, y_pred):
    SS_res =  np.sum(np.square(y_true - y_pred))
    SS_tot = np.sum(np.square(y_true - np.mean(y_true)))
    r2_mine=( 1 - SS_res/(SS_tot + keras.backend.epsilon()) )
    return ( 1 - SS_res/(SS_tot + keras.backend.epsilon()) )

global report_val_acc 
global report_val_r2
global report_val_mse
report_val_acc=[]
report_val_r2=[]
report_val_mse=[]
# LOG FILE
global log_file

"""
DATA GENERATORS CREATION
"""
train_generator=DataGenerator(data_list, concept=CONCEPT, batch_size=BATCH_SIZE, data_type=0)
train_generator2=DataGenerator(data_list, concept=CONCEPT, batch_size=BATCH_SIZE, data_type=1)#get_batch_data(data_list, batch_size=BATCH_SIZE)
val_generator=DataGenerator(val_list, concept=CONCEPT, batch_size=BATCH_SIZE, data_type=1) #get_test_batch(val_list, batch_size=BATCH_SIZE)
val_generator2= DataGenerator(val_list, concept=CONCEPT, batch_size=BATCH_SIZE, data_type=1) #get_test_batch(val_list, batch_size=BATCH_SIZE)
test_generator= DataGenerator(test_list, concept=CONCEPT, batch_size=BATCH_SIZE, data_type=1) #get_test_batch(test_list, batch_size=BATCH_SIZE)

""" 
Get trainable model with Hepistemic Uncertainty Weighted Loss 
"""

def get_trainable_model(baseline_model):
    inp = keras.layers.Input(shape=(224,224,3,), name='inp')
    y1_pred, y2_pred = baseline_model(inp)
    y1_true=keras.layers.Input(shape=(1,),name='y1_true')
    #y2_true=keras.layers.Input(shape=(7,),name='y2_true')
    out = CustomMultiLossLayer(nb_outputs=2, new_folder=new_folder)([y1_true, y2_true, y1_pred, y2_pred])
    return Model(input=[inp, y1_true, y2_true], output=out)

def get_trainable_model(baseline_model):
    inp = keras.layers.Input(shape=(224,224,3,), name='inp')
    outputs = baseline_model(inp)
    n_extra_concepts = len(outputs) -1
    print(n_extra_concepts)
    y_true=keras.layers.Input(shape=(1,),name='y_true')
    #domain_true=keras.layers.Input(shape=(7,),name='domain_true')
    extra_concepts_true=[]
    for i in range(n_extra_concepts):
        print('extra_{}'.format(i))
        extra_true=keras.layers.Input(shape=(1,), name='extra_{}'.format(i))
        extra_concepts_true.append(extra_true)
    new_model_input=[inp, y_true]#, domain_true]
    loss_inputs=[y_true]#, domain_true]
    for i in range(len(extra_concepts_true)):
        new_model_input.append(extra_concepts_true[i])
        loss_inputs.append(extra_concepts_true[i])
    for out_ in outputs:
        loss_inputs.append(out_)
    out = CustomMultiLossLayer(nb_outputs=len(outputs), new_folder=new_folder)(loss_inputs)
    return Model(input=new_model_input, output=out)

verbose=True
def custom_train_model(hu_model,
                       epochs= 1, 
                       lr=1e-4, verbose=True,
                       train_generator=train_generator,
                       val_generator=val_generator,
                       grl_layer=None,
                      ):
    opt = keras.optimizers.SGD(lr=lr, momentum=0.9, nesterov=True)
    compile_model(hu_model,opt,loss=None,metrics=None)
    history = hu_model.fit_generator(train_generator,
                    steps_per_epoch= len(data_list) // (BATCH_SIZE ),
                    callbacks=callbacks,
                    epochs=epochs,
                    verbose=verbose,
                    workers=4,
                    use_multiprocessing=False,
                    validation_data= val_generator,
                    validation_steps= len(val_list)//BATCH_SIZE) 
    return hu_model

"""
Training 
"""
starting_time = time.time()
print("training")
print("phase 1")
main_task_weight=1. 
domain_weight = 1.
model = get_baseline_model(hp_lambda=0., c_list=c_list)
t_m = get_trainable_model(model)
t_m = custom_train_model(t_m, epochs=80,
                         lr=1e-4, grl_layer=None)
base_model_weights=model.get_weights()
print("phase 2")
main_task_weight=1.
domain_weight=1. 
model=get_baseline_model(hp_lambda=1., c_list=c_list)
model.set_weights(base_model_weights)
t_m_ = get_trainable_model(model)
t_m = custom_train_model(t_m_, epochs=20, 
                         lr=1e-4, grl_layer=None)
"""
# 20 epochs: gradient reversal
# not run yet
#print("phase 3")
#main_task_weight=1. 
#domain_weight = 1e-2
#model = get_baseline_model(hp_lambda=-1.)
#t_m = get_trainable_model(model)
#t_m = custom_train_model(t_m, epochs=5s, lr=1e-4, grl_layer=model.grl_layer)
#base_model_weights=model.get_weights()

main_task_weight=1.
domain_weight=1. #0. #(higher learning rate is needed for this task)
model=get_baseline_model(hp_lambda=10.)
model.set_weights(base_model_weights)
#model.compile(opt)
t_m_ = get_trainable_model(model)
t_m = custom_train_model(t_m_,epochs=5, lr=1e-4, grl_layer=model.grl_layer)
# We run some more iterations on the main task
main_task_weight=1. 
domain_weight = 1
model=get_baseline_model(hp_lambda=10.)
model.set_weights(base_model_weights)
#model.compile(opt)
t_m = get_trainable_model(model)
t_m = custom_train_model(t_m, epochs=2, lr=1e-4)
"""
# end of scheduling 
end_time=time.time()
total_training_time = end_time - starting_time 
#
#print(history.history.keys())
log_file=open("{}/{}_log.txt".format(new_folder, EXPERIMENT_TYPE), 'a')
log_file.write('Time elapsed for model training: {}'.format(total_training_time))
log_file.close()
#np.save('{}/training_log'.format(new_folder), history.history)
np.save('{}/val_acc_log'.format(new_folder), report_val_acc)
np.save('{}/val_r2_log'.format(new_folder), report_val_r2)
print("++++++ HEPISTEMIC UNCERTAINTY WEIGHTED LOSS TRAINING: OK WE'RE DONE OVER HERE ++++++ FOLDER: {} ++++++ GOOD JOB. ".format(new_folder))
exit(0)


# In[ ]:
'''This class is written to take a directory containing images and output predictions in the format
of a pandas dataframe. The methods can be used to first instantiate the Use_Net instance. The Get_Data method
is given a directory, as well as the micron/pixel density of the image and returns codes computed by ResNet50.
At this point, one can run the Predict Method to create two dataframes. df1 contains the actual call while df2 contains
the p-value of that call.'''
from utils import Get_Bottleneck_Tiles_Unlabeled
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import pickle
import pandas as pd

class Use_Net():

    def __init__(self, Instance_Name='Net'):
        self.Instance_Name = Instance_Name

    def Get_Data(self,directory, Load_Prev_Data=False ,mpratio=0.42):
        self.Load_Prev_Data = Load_Prev_Data

        if self.Load_Prev_Data is False:

            self.codes, self.files = Get_Bottleneck_Tiles_Unlabeled(directory, mpratio)

            with open(self.Instance_Name + '.pkl', 'wb') as f:
                pickle.dump(self, f)

        else:
            with open(self.Instance_Name + '.pkl', 'rb') as f:
                self.__dict__.update(pickle.load(f).__dict__)

    def Predict(self):
        # Load Weights from Trained Model
        with open('Weights.pkl', 'rb') as f:
            self.fc1_bias_final, self.fc1_kernel_final, self.logits_bias_final, self.logits_kernel_final,self.lb = pickle.load(f)

        with tf.Session() as sess:
            saver = tf.train.import_meta_graph('Image_Aug_A/Image_Aug_A.ckpt.meta')
            saver.restore(sess, tf.train.latest_checkpoint('Image_Aug_A'))
            graph = tf.get_default_graph()
            inputs_ = graph.get_tensor_by_name('Input:0')
            logits_kernel = graph.get_tensor_by_name('logits/kernel:0')
            logits_bias = graph.get_tensor_by_name('logits/bias:0')
            fc_kernel = graph.get_tensor_by_name('fc1/kernel:0')
            fc_bias = graph.get_tensor_by_name('fc1/bias:0')
            tf.assign(logits_kernel, self.logits_kernel_final)
            tf.assign(logits_bias, self.logits_bias_final)
            tf.assign(fc_kernel,self.fc1_kernel_final)
            tf.assign(fc_bias,self.fc1_bias_final)
            predicted = graph.get_tensor_by_name('predicted:0')
            feed_dict = {inputs_: self.codes}
            self.predicted = sess.run(predicted, feed_dict=feed_dict)
            encoder=OneHotEncoder(sparse=False)
            encoder.fit(np.asarray([0,1,2,3]).reshape(-1,1))
            self.call=self.lb.inverse_transform(encoder.transform(np.argmax(self.predicted,axis=1).reshape(-1,1)))
            df=pd.DataFrame()
            df['File']=self.files
            df['Prediction']=self.call

            df2=pd.DataFrame()
            df2['File'] = self.files
            classes=self.lb.classes_

            for ii,type in enumerate(classes,0):
                df2[type]=self.predicted[:,ii]

            self.df1=df
            self.df2=df2

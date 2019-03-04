######## IMPORT LIBRARIES ###############
import warnings
warnings.filterwarnings('ignore')
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
import cv2
import imgaug as ia
from imgaug import augmenters as iaa
import argparse
from PIL import Image
import numpy as np
import pandas as pd
import multiprocessing
from multiprocessing import Pool
import random
import time
import math

transformation_list = ['add', 'add_hue_saturation', 'agn', 'blur', 'sharpen', 'grayscale', 'contrast_norm']

img_file =  '../data/train.txt'
aug_path =  '../data/images_aug_8k'

output_file = '../data/train_aug_8k.txt'
image_path = '../data/train_images/'

if not os.path.exists(aug_path):
    os.makedirs(aug_path)

images_column_name = "local_path"
label_column_name = "labels"
total_trans = len(transformation_list)
total_aug_cnt = 14
trans_repeat_factor = 2


df = pd.read_csv(img_file, header = None, sep = " ")
n_classes = df[1].nunique()
print ('total unique classes '  + str(n_classes) )
max_image_class = 8000

def get_unique_label_counts(df):
    
    return df[label_column_name].value_counts().to_dict()

def augment_count_calculate(label_count_dict, total_augments, max_count):
    
    augment_count_dict = {} 
    
    for key, value in label_count_dict.items():

        if value >= max_count:
            augment_count_dict[key] = max_count - value, 0
            
        else: 
            if value * (total_augments + 1) >= max_count:
                augment_count_dict[key] = max_count % value, int(max_count/value) - 1 
            else:
                augment_count_dict[key] = 0, total_augments
                
    return augment_count_dict


def prepare_final_aug_df(df,label_count_dict, augment_count_dict):
    
    new_rows_for_augment =  pd.DataFrame(columns = df.columns)
    for key, value in augment_count_dict.items():

        #checking for downsample, no need of augment
        image_sample, aug_count = value
        
        if image_sample < 0: 
            print ('no augmentation required')
#             print("dropping rows for " + str(key) + " as it has extra " + str(abs(aug_count))+ \
#                   " rows than which are possible with augmentation.")  
#             random.seed(1)
#             df = df.drop( random.sample( list(df[df[label_column_name] == key].index), abs(image_sample) ) )
   
        else:
            label_dataframe = df[df[label_column_name] == key].reset_index(drop = True)        
            np.random.seed(1)
            rows = np.random.choice(label_dataframe.index, image_sample, replace=False)
            label_dataframe['Aug_cnt'] = aug_count
            label_dataframe['Aug_cnt'].loc[rows] = label_dataframe['Aug_cnt'].loc[rows] + 1
            label_dataframe = label_dataframe[label_dataframe.Aug_cnt > 0] 
            new_rows_for_augment =  new_rows_for_augment.append(label_dataframe, ignore_index = True)       
     
    return new_rows_for_augment, df
    
########################################################################

cores = multiprocessing.cpu_count()
def parallelize_dataframe(df, func):
    df_split = np.array_split(df, cores)
    pool = Pool(cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

        
def augment(df):
    """
    The target method that the process uses to augment the specified image
    """
    df_trans = pd.DataFrame(columns = df.columns)
    
    add = iaa.Add((-10, 10), per_channel=0.5)
    add_hue_saturation = iaa.AddToHueAndSaturation((-20, 20))
    agn = iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.04*255), per_channel=0.5)
    blur = iaa.GaussianBlur(sigma=(0, 1.0))
    sharpen= iaa.Sharpen(alpha=(0.1, 0.6), lightness=(0.75, 1.5))
    contrast_norm = iaa.ContrastNormalization((0.5,1.5), per_channel=0.5) 
    grayscale = iaa.Grayscale(alpha=(0.1, 0.8))
    
    images_to_augment = int(df['Aug_cnt'].sum())
    print ('Total Images to Augment on a core: ' + str(images_to_augment) )
    for i, (img, label, aug_count) in enumerate( zip( df['local_path'], df[label_column_name], df['Aug_cnt'])):
        
        augmented_images_perc = int(i * 100/images_to_augment)
        if i % int(images_to_augment*0.2) == 0 and i > 0:
            print ('{}% images augmented'.format(augmented_images_perc) )
            
        if i % 20 == 0 and i > 0:
            print ("augmenting {}th image".format(i) )
            
        try:
            im = Image.open(img)
            im_ar = np.array(im)
            trans_required = random.sample(transformation_list * trans_repeat_factor, int(aug_count) )
            trans_required = [x + '_' + str(i) for i,x in enumerate(trans_required)]
            print ('trans_required: ' , trans_required )   
            
            for trans in trans_required:      
                if trans.strip().startswith("add"):
                    im_trans = Image.fromarray(add.augment_image(im_ar))
                    
                elif trans.strip().startswith("add_hue_saturation"):
                    im_trans = Image.fromarray(add_hue_saturation.augment_image(im_ar))
                    
                elif trans.strip().startswith("agn"):
                    im_trans = Image.fromarray(agn.augment_image(im_ar))
                    
                elif trans.strip().startswith("blur"):
                    im_trans = Image.fromarray(blur.augment_image(im_ar))
                    
                elif trans.strip().startswith("sharpen"):
                    im_trans = Image.fromarray(sharpen.augment_image(im_ar))
                    
                elif trans.strip().startswith("contrast_norm"):
                    im_trans = Image.fromarray(contrast_norm.augment_image(im_ar))
                    
                elif trans.strip().startswith("grayscale"):
                    im_trans = Image.fromarray(grayscale.augment_image(im_ar))

                else:
                    continue
                
                image_id = img.split('/')[-1]
                temp_aug_path = aug_path + "/" + trans + '_' + image_id
                im_trans.save(temp_aug_path)
                df1 = df.iloc[i,:-1]  # taking i th row only
                df1['local_path'] = temp_aug_path
                df1[label_column_name] = label
                df_trans = df_trans.append(df1, ignore_index = True)

        except Exception as e:      
            
            print (i, img + " not a valid image" )
            print ("Exception is :"+str(e))
     
    ## Dropping Augmentation count column
    df_trans = df_trans[['local_path', label_column_name]]
    return df_trans

    
def main():
    #reading input file
    print("reading data input file ")
    df = pd.read_csv(img_file, header= None, sep = ' ')
    column_name_list = []
    column_name_list.append(images_column_name)
    column_name_list.append(label_column_name)
    print("column names : ", column_name_list)
    df.columns = column_name_list
    
    print("\nlength of input dataframe : " + str(len(df.index)))
                  
    print("\nreading label counts")
    label_count_dict = get_unique_label_counts(df)
    print("\nlabel count is as : " + str(label_count_dict))
    
    augment_count_dict = augment_count_calculate(label_count_dict, total_augments = total_aug_cnt, max_count = max_image_class)
    print("\naugment counts for labels : " + str(augment_count_dict))
        
    rows_for_augment, df = prepare_final_aug_df(df,label_count_dict, augment_count_dict)
    images_to_augment = int (rows_for_augment['Aug_cnt'].sum())
    print ('\nTotal Images to Augment: ' + str(images_to_augment))
    start = time.time()
    augmented_df = parallelize_dataframe(rows_for_augment, augment)
    # augmented_df = augment(rows_for_augment)
    print ('\ntime taken in seconds .... ' + str(time.time()-start) )
    
    print('Unique labels in original df: ' + str(df[label_column_name].unique()) )
    print('Unique labels in augmented df: ' + str(df[label_column_name].unique()) )
    
    print("\nlength of augmented dataframe : " + str(len(augmented_df.index)))
    print("length of original dataframe : " + str(len(df.index)))
    print ('\nAppending augmented dataframe with original dataframe...')
    df = df.append(augmented_df, ignore_index = True)
    print("saving final file after augment")
    df.to_csv(output_file, header = False, sep = ' ', index = False)
    print("length of final dataframe : " + str(len(df.index)))
    print("\ndone")

main()

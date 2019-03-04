import pandas as pd
import numpy as np
from ast import literal_eval

# df1 = pd.read_csv('../submissions/aug8k_densenet121_prob_20190107011043.csv', usecols= ['id', 'Probablity'])
df2 = pd.read_csv('../submissions/aug8k_res50_prob_20190107005605.csv', usecols= ['id', 'Probablity']) 
df3 = pd.read_csv('../submissions/aug8k_inception_v3_prob_20190107014357.csv', usecols= ['id', 'Probablity']) 
# df = df1.merge(df2, how = 'left', on = 'id')
df = df2.merge(df3, how = 'left', on = 'id')

prob_cols = [col for col in df.columns if 'Prob' in col]
for col in [col for col in df.columns if 'Prob' in col]:
    df[col] =  [ np.array(x.replace('[','').replace(']','').replace('\n', '').\
                             split(' ') )  for x in df[col] ]
    df[col] = [ [float(x) for x in list(y) if x != ''] for y in df[col]]
    
# df['mean_prob'] = [ np.array(list((sum(x)/len(prob_cols) for x in zip(y,z,q)))) \
#                    for y,z,q in zip(df[prob_cols[0]], df[prob_cols[1]], df[prob_cols[2]])]  #,

df['mean_prob'] = [ np.array(list((sum(x)/len(prob_cols) for x in zip(y,z)))) \
                   for y,z in zip(df[prob_cols[0]], df[prob_cols[1]])]  #,


df['category'] = [ x.argmax() for x in df.mean_prob]

submission = df[['id', 'category']]
submission.category = submission.category + 1

# filename = f'ensemble_resnet50_densenet121_inseptionv3_submission.csv'
filename = f'ensemble_aug8k_resnet50_inceptionv3_submission.csv'
output_folder = '../submissions'
submission.to_csv(f'{output_folder}/{filename}', index=False)



{\rtf1\ansi\ansicpg1252\cocoartf1504\cocoasubrtf830
{\fonttbl\f0\fswiss\fcharset0 ArialMT;\f1\froman\fcharset0 Times-Roman;}
{\colortbl;\red255\green255\blue255;\red0\green0\blue0;}
{\*\expandedcolortbl;;\cssrgb\c0\c0\c0;}
{\*\listtable{\list\listtemplateid1\listhybrid{\listlevel\levelnfc0\levelnfcn0\leveljc0\leveljcn0\levelfollow0\levelstartat1\levelspace360\levelindent0{\*\levelmarker \{decimal\}.}{\leveltext\leveltemplateid1\'02\'00.;}{\levelnumbers\'01;}\fi-360\li720\lin720 }{\listname ;}\listid1}}
{\*\listoverridetable{\listoverride\listid1\listoverridecount0\ls1}}
\paperw11900\paperh16840\margl1440\margr1440\vieww28300\viewh17700\viewkind0
\deftab720
\pard\pardeftab720\sl660\partightenfactor0

\f0\b\fs48 \cf2 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 How to reproduce results
\f1\b0\fs24 \
\pard\pardeftab720\sl280\partightenfactor0
\cf2 \
\pard\tx220\tx720\pardeftab720\li720\fi-720\sl400\partightenfactor0
\ls1\ilvl0
\f0\fs29\fsmilli14667 \cf2 \kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	1.	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 After downloading the images, keep them in separate sub-folders \'91train_images\'92 and \'91test_images\'92 in \'91data\'92 folder\uc0\u8232 \
\ls1\ilvl0\kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	2.	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 Run 
\b data_prep.ipynb
\b0  code to create 
\b train.txt
\b0  and 
\b valid.txt
\b0  files.\uc0\u8232 \
\ls1\ilvl0\kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	3.	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 Run 
\b augment.py
\b0  code to create 
\b train_aug_8k.txt
\b0  file. This file was used for all the training.\uc0\u8232 \
\ls1\ilvl0\kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	4.	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 Run 
\b model_resnet.py, model_inception.py 
\b0 and
\b  model_densenet.py
\b0  code to generate trained models for resnet-50, inception-V3 and densenet-121. All models are validated on valid.txt file.\uc0\u8232 \
\ls1\ilvl0\kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	5.	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 Run 
\b predict.py
\b0  code to create prediction for each of the aforementioned models. This code will generate the predictions as well as the probability for each category.\uc0\u8232 \
\ls1\ilvl0\kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	6.	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 Finally run 
\b ensemble.py
\b0  code to create final submission file which is ensemble of resnet-50 and inception-V3.\uc0\u8232 \
}
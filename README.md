# Retinal-Pathology-Classifier
Retinal Pathology Classification

## Data for this work is taken from a Kaggle contest: https://www.kaggle.com/c/vietai-advance-course-retinal-disease-detection/overview
Explanation of the data set:

The training data set contains 3435 retinal images that represent multiple pathological disorders. The patholgy classes and corresponding labels are: included in 'train.csv' file and each image can have more than one class category (multiple pathologies).
The labels for each image are

```
-opacity (0), 
-diabetic retinopathy (1), 
-glaucoma (2),
-macular edema (3),
-macular degeneration (4),
-retinal vascular occlusion (5)
-normal (6)
```
The test data set contains 350 unlabelled images.

Overview
Color-filtered fundus images visualize the rear of an eye called retina (Figure 1). Fundus image provides doctors with a snapshot on the interior of the eye of patients. Based on this type of image, doctor will be able to read abnormalities present on the back of the eye, thus making diagnosis easier and more accurate. Many eye diseases can be found using fundus images, such as diabetic retinopathy, glaucoma, and macular degeneration.
Current available public datasets (EYEPACS, Messidor, etc.), although rich in quantity, only focus on diabetic retinopathy. However, more often than not, a patient can have two or more diseases concurrently. To overcome these 2 problems, we take other common diseases into consideration and create a multi-labeled dataset. The following will describe the dataset in details.


Problem description
The dataset includes 3,285 images from CTEH (3.210 abnormals and 75 normals) and 500 normal images from Messidor and EYEPACS dataset. The abnormalities include: opacity, diabetic retinopathy, glaucoma, macular edema, macular degeneration, and retinal vascular occlusion.

On this work we will focus on glaucoma and diabetic retinopathy vs. normal only. 

Acknowledgements
We thank AI Department – Cao Thang Eye Hospital (CTEH) for providing this dataset

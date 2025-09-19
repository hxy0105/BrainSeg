import SimpleITK as sitk
import os
import glob
import numpy as np


rows = 256 
cols = 256


def label2class(label):
    x = np.zeros([rows, cols, 4])
    for i in range(rows):
        for j in range(cols):
            x[i, j, int(label[i][j])] = 1
    return x


current = os.getcwd()
GT_path = os.path.join(current, "dataset", "GT") + os.sep
GTs = glob.glob(os.path.join(GT_path, '*.nii'))


#search for all NIFT format images
Org_path = os.path.join(current, "dataset", "Org") + os.sep    
Orgs = glob.glob(os.path.join(Org_path, '*.nii'))
print(Orgs)

subjects = 20
slices = 2294
GT_datas = np.ndarray((slices,rows,cols,4), dtype='uint16')
Org_datas = np.ndarray((slices,rows,cols,1), dtype='uint16')
i = 0
x = 0   
for f in GTs[0:subjects]:
    x += 1
    image = sitk.ReadImage(f)
    GT = sitk.GetArrayFromImage(image)
    GT = GT.reshape(GT.shape[0],GT.shape[1],GT.shape[2])
    
    for j in range(1,GT.shape[0],1): 
        ROI = GT[j,:,:]
        if GT.shape[1] < 256:  #zero padding to (256,256)
            ROI = np.pad(ROI,(((256-GT.shape[1])//2,(256-GT.shape[1])//2),(0,0)),'constant', constant_values=(0, 0))
        if GT.shape[2] < 256:
            ROI = np.pad(ROI,((0,0),((256-GT.shape[2])//2,(256-GT.shape[2])//2)),'constant', constant_values=(0, 0))
        #print(ROI.shape)
        ROI = label2class(ROI)
        GT_datas[i] = ROI
        i += 1
            
i = 0
x = 0
for f in Orgs[0:subjects]:
    x += 1
    image = sitk.ReadImage(f)
    Org = sitk.GetArrayFromImage(image)
    Org = Org.reshape(Org.shape[0],Org.shape[1],Org.shape[2])
    #print(Org.shape)
    for j in range(1,Org.shape[0],1): # extract slices starting from 10 and ending at 152 with interval of 3 slices in between
        ROI = Org[j,:,:]       
        if Org.shape[1] < 256:  #zero padding to (256,256)
            ROI = np.pad(ROI,(((256-Org.shape[1])//2,(256-Org.shape[1])//2),(0,0)),'constant', constant_values=(0, 0))
        if Org.shape[2] < 256:
            ROI = np.pad(ROI,((0,0),((256-Org.shape[2])//2,(256-Org.shape[2])//2)),'constant', constant_values=(0, 0))
        #print(ROI.shape)
        ROI = ROI.reshape(ROI.shape[0], ROI.shape[1], 1)
        Org_datas[i] = ROI
        i += 1
            

Org_datas = Org_datas/Org_datas.max()

# # #shuffle
# indices = np.random.permutation(GT_datas.shape[0])
# GT_datas = GT_datas[indices]
# Org_datas = Org_datas[indices]

#train data and test data
GT_train_data = GT_datas[0:1794,:,:]
Org_train_data = Org_datas[0:1794,:,:]

GT_test_data = GT_datas[1794:2294,:,:]
Org_test_data = Org_datas[1794:2294,:,:]


# #save as numpy dataset
np.save('GT_train_data_ns.npy', GT_train_data)
np.save('Org_train_data_ns.npy', Org_train_data)
np.save('GT_test_data.npy_ns', GT_test_data)
np.save('Org_test_data.npy_ns', Org_test_data)

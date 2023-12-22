import os, pdb
from glob import glob
from Computer_Vision_Functions import Horizontal_Flip, Image_Rotation, Mask_Rotation, Hausdorff_Distance, DICE_Coeff

#Run Augmentations on the Original Images and the Masks
Augmentations_Both = input('Do you want to perform augmentations on the images and the masks? Yes = 1, No =  0 ')
Augmentations_Both = bool(eval(Augmentations_Both))

if Augmentations_Both is True:
    #img_path = '../imgs'
    img_path  = input('Please input the relative location of the original image folder (defualt: ../imgs): ')
    #mask_path = '../masks'
    mask_path = input('Please input the relative location of the original mask folder (default: ../masks): ')

    Flip = input('Do you want to flip the Images and/or Masks Horizontally? Yes = 1, No = 0: ')
    Flip = bool(eval(Flip))

    if Flip is True:
        img_flip_path='../imgs_flipped/'
        mask_flip_path = '../masks_flipped/'

        imgExists = os.path.exists(img_flip_path)
        maskExists = os.path.exists(mask_flip_path)
        if imgExists is False:
            os.makedirs(img_flip_path)
        else:
            img_flip_path = input('The default folder already exists! Specify a relative path for the flipped image folder (default syntax: ../folder_name/): ')
            os.makedirs(img_flip_path)
        if maskExists is False:
            os.makedirs(mask_flip_path)
        else:
            mask_flip_path = input('The default folder already exists! Specify a relative path for the flipped mask folder in the following format (default syntax: ../folder_name/): ')
            os.makedirs(mask_flip_path)

    Rotate = input('Do you want to rotate your images and masks? Yes = 1, No = 0: ')
    Rotate = bool(eval(Rotate))

    if Rotate is True:
        Degree = input('Input a rotation degree in between [-180 - 180]: ')
        Degree = float(Degree)
        img_rotation_path='../imgs_rotated/'
        mask_rotation_path = '../masks_rotated/'

        imgExists = os.path.exists(img_rotation_path)
        maskExists = os.path.exists(mask_rotation_path)
        if imgExists is False:
            os.makedirs(img_rotation_path)
        else:
            img_rotation_path = input('The default folder already exists! Specify a relative path for the rotated image format (default syntax: ../folder_name/): ')
            os.makedirs(img_rotation_path)
        if maskExists is False:
            os.makedirs(mask_rotation_path)
        else:
            mask_rotation_path = input('The default folder already exists! Specify a relative path for the rotated mask format (default syntax: ../folder_name/): ')
            os.makedirs(mask_rotation_path)

        FlipRot = input('Do you want to flip the rotated images and masks? Yes = 1, No = 0: ')
        FlipRot = bool(eval(Rotate))

        if FlipRot is True:
            img_RotFlip_path='../imgs_rotated_flipped/'
            mask_RotFlip_path = '../masks_rotated_flipped/'

            RotFlipExists = os.path.exists(img_RotFlip_path)
            RotFlipmaskExists = os.path.exists(mask_RotFlip_path)

            if RotFlipExists is False:
                os.makedirs(img_RotFlip_path)
            else:
                img_RotFlip_path = input('The default folder already exists! Specify a relative path for the rotated image format (default syntax: ../folder_name/): ')
                os.makedirs(img_RotFlip_path)
            if RotFlipmaskExists is False:
                os.makedirs(mask_RotFlip_path)
            else:
                mask_RotFlip_path = input('The default folder already exists! Specify a relative path for the rotated mask format (default syntax: ../folder_name/): ')
                os.makedirs(mask_RotFlip_path)
        
    #img_file_extension = 'jpg'
    #mask_file_extension = 'gif'
    #pred_file_extension = 'gif'

    img_file_extension = input('Please input the file extension of the images (default:jpg): ')
    mask_file_extension = input('Please input the file extension of the masks (default:gif): ')

    image_list = glob(os.path.join(img_path,"*.{:s}").format(img_file_extension))
    mask_list = glob(os.path.join(mask_path,"*.{:s}").format(mask_file_extension))

    for img_path,mask_path in zip(image_list,mask_list):
        #Finish organizing this function and set up the 
        if Flip is True:
            Horizontal_Flip(img_path,img_flip_path)    
            Horizontal_Flip(mask_path,mask_flip_path)
        if Rotate is True:
            Image_Rotation(img_path,Degree,img_rotation_path)
            Mask_Rotation(mask_path,Degree,mask_rotation_path)
            if FlipRot is True:
                Horizontal_Flip(img_rotation_path,img_RotFlip_path)    
                Horizontal_Flip(mask_rotation_path,mask_RotFlip_path)

Augmentations_Imgs = input('Do you want to perform augmentations only on the images? Yes = 1, No = 0: ')
Augmentations_Imgs = bool(eval(Augmentations_Imgs))

if Augmentations_Imgs is True:
    #img_path = '../imgs'
    img_path  = input('Please input the relative location of the original image folder (defualt: ../imgs): ')

    Flip = input('Do you want to flip the Images Horizontally? Yes = 1, No = 0: ')
    Flip = bool(eval(Flip))

    if Flip is True:
        img_flip_path='../imgs_flipped/'

        imgExists = os.path.exists(img_flip_path)
        if imgExists is False:
            os.makedirs(img_flip_path)
        else:
            img_flip_path = input('The default folder already exists! Specify a relative path for the flipped image folder (default syntax: ../folder_name/): ')
            os.makedirs(img_flip_path)

    Rotate = input('Do you want to rotate your images? Yes = 1, No = 0: ')
    Rotate = bool(eval(Rotate))

    if Rotate is True:
        Degree = input('Input a rotation degree in between [-180 - 180]: ')
        Degree = float(Degree)
        img_rotation_path='../imgs_rotated/'

        imgExists = os.path.exists(img_rotation_path)
        if imgExists is False:
            os.makedirs(img_rotation_path)
        else:
            img_rotation_path = input('The default folder already exists! Specify a relative path for the rotated image format (default syntax: ../folder_name/): ')
            os.makedirs(img_rotation_path)


        FlipRot = input('Do you want to flip the rotated images? Yes = 1, No = 0: ')
        FlipRot = bool(eval(Rotate))

        if FlipRot is True:
            img_RotFlip_path='../imgs_rotated_flipped/'

            RotFlipExists = os.path.exists(img_RotFlip_path)
            if RotFlipExists is False:
                os.makedirs(img_RotFlip_path)
            else:
                img_RotFlip_path = input('The default folder already exists! Specify a relative path for the rotated image format (default syntax: ../folder_name/): ')
                os.makedirs(img_RotFlip_path)
        
    #img_file_extension = 'jpg'
    img_file_extension = input('Please input the file extension of the images (default:jpg): ')

    image_list = glob(os.path.join(img_path,"*.{:s}").format(img_file_extension))

    for img_path in image_list:
        #Finish organizing this function and set up the 
        if Flip is True:
            Horizontal_Flip(img_path,img_flip_path)    
        if Rotate is True:
            Image_Rotation(img_path,Degree,img_rotation_path)
            if FlipRot is True:
                Horizontal_Flip(img_rotation_path,img_RotFlip_path)    

Run_Scores = input('Do you want to perform DICE and Hausdorff Distance Calculations? Yes = 1, No = 0: ')
Run_Scores = bool(eval(Run_Scores))

if Run_Scores is True:
#Run Scores For all the Predictions
    HD_Scores = []
    DICE_Scores = []

    #mask_path = '../masks'
    mask_path = input('Please input the relative location of the original mask folder (default: ../masks): ')
    #pred_path = '../predictions'
    pred_path = input('Please input the relative location of the network prediction mask folder: ')
    
    #mask_file_extension = input('Please input the file extension of the masks (default:gif): ')
    #pred_file_extension = input('Please input the file extension of the predictions (default:gif): ')

    def list_files_in_folder(folder_path):
        #files_list = []
        try:
            # Get the list of files in the folder
            files = os.listdir(folder_path)

            # Print the list of files
            #for file in files:
            #    files_list.append(file)

            return files


        except Exception as e:
            print(f"An error occurred: {e}")
        
        
    mask_list = list_files_in_folder(mask_path)
    pred_list = list_files_in_folder(pred_path)

    mask_list.sort()
    pred_list.sort()
    
    #mask_list = glob(os.path.join(mask_path,"*.{:s}").format(mask_file_extension))
    #pred_list = glob(os.path.join(pred_path,"*.{:s}").format(pred_file_extension))

    for mask, pred in zip(mask_list,pred_list):
        mask = os.path.join(mask_path, mask)
        pred = os.path.join(pred_path, pred)
        Distance,_,_ = Hausdorff_Distance(mask,pred)
        DICE_Score = DICE_Coeff(mask,pred)
        HD_Scores.append(Distance)
        DICE_Scores.append(DICE_Score)
    
    HD_Scores.sort()
    DICE_Scores.sort()
    Average_HD = sum(HD_Scores)/len(HD_Scores)
    Average_DICE = sum(DICE_Scores)/len(DICE_Scores)

    #pdb.set_trace()
    
    print('The maximum Hausdorff Distance in the dataset is: ' + str(HD_Scores[-1]) )
    print('The minimum Hausdorff Distance in the dataset is: ' + str(HD_Scores[0]))
    print('The average Hausdorff Distance in the dataset is: ' + str(Average_HD))

    print('The maximum DICE Score in the dataset is: ' + str(DICE_Scores[-1] ))
    print('The minimum DICE Score in the dataset is: ' + str(DICE_Scores[0]))
    print('The average DICE Score in the dataset is: ' + str(Average_DICE))
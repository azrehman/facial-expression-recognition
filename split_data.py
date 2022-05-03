import splitfolders

input_dir = 'face_images'



#split into 80/10/10 train/val/test setss
#splitfolders.ratio(input_dir, output='face_images_80_10_10',
#    seed=1337, ratio=(.8, .1, .1), group_prefix=None, move=False) # default values

#split into 80/10/10 train/val/test setss
splitfolders.ratio('radiate_faces', output='radiate_faces_80_10_10',
    seed=1337, ratio=(.8, .1, .1), group_prefix=None, move=False) # default values

import splitfolders

input_dir = 'face_images'


# split into 80/20 train/val sets
#splitfolders.ratio(input_dir, output='face_images_80_20',
#    seed=1337, ratio=(.8, .2), group_prefix=None, move=False) # default values

# split into 90/10 train/val sets
#splitfolders.ratio(input_dir, output='face_images_90_10',
#    seed=1337, ratio=(.9, .1), group_prefix=None, move=False) # default values

# split into 80/10/10 train/val/test setss
#splitfolders.ratio(input_dir, output='face_images_80_10_10',
#    seed=1337, ratio=(.8, .1, .1), group_prefix=None, move=False) # default values

# split into 70/10/20 train/val/test setss
splitfolders.ratio(input_dir, output='face_images_70_10_20',
    seed=1337, ratio=(.7, .1, .2), group_prefix=None, move=False) # default values

# split into 50/10/40 train/val/test setss
splitfolders.ratio(input_dir, output='face_images_50_10_40',
    seed=1337, ratio=(.5, .1, .4), group_prefix=None, move=False) # default values

# split into 20/10/70 train/val/test setss
splitfolders.ratio(input_dir, output='face_images_20_10_70',
    seed=1337, ratio=(.2, .1, .7), group_prefix=None, move=False) # default values

# split into 10/10/80 train/val/test setss
splitfolders.ratio(input_dir, output='face_images_10_10_80',
    seed=1337, ratio=(.1, .1, .8), group_prefix=None, move=False) # default values

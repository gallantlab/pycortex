===================================
Surface Segmentation and Flattening
===================================


Welcome! This is a guide for the full process of making flatmaps, which allow us to visualize brain data in a more intuitive way than voxelized 3D images of brain data. There are three main phases of the process:

**1. Segmentation**

Freesurfer will automatically discern where white matter and gray matter are in the brain. The software is good but not perfect, so you will need to look through its results and fine-tune the surfaces it generates so we can generate a 3D model of the brain.


**2. Cutting and flattening**

The brain model is exported to a 3D modeling program called Blender, where you will make the cuts necessary to transform a 3D object into a flattened map with as little warping as possible.


**3. Labeling ROIs**

Here you will project functional data (semantic betas or localizer data, as well as retinotopic) onto the flatmaps, allowing you to label Regions of Interest on the brain - areas responsive to faces, scenes, or whatever else we’re analyzing.

In this guide, we will go over the first two steps.




**Unpacking Data, Importing data, & Starting Freesurfer Unpacking Data**

If you have raw data, you want to unpack the .tar back to multiple dcm files. To do this, into terminal type:

       ``tar xvf filename.tar``


If possible, you want to proceed with MEMPRAGE RMS data.




Starting & Setting up Freesurfer
##################################


To open freesurfer:
    ``source_directory_name_here/SetUpFreeSurfer.sh``

For example:
    ``/auto/myfolder/freesurfer/SetUpFreeSurfer.sh``

Create a “subjects” directory. If a "subjects" directory doesn't exist, make one in the FreeSurfer directory. Freesurfer is finicky about directories, so this step is crucial.




Importing Data Into Freesurfer
#################################

After you've unpacked the raw data, there are 2 ways to choose from to import your data to into freesurfer:

**1. Using .dcm files:** To do this, go into the directory of the dicom files you want to use and then type:
    ``recon-all -i ./NameofFirstDicomFile.dcm -s <subject_name>``
For example:
    ``recon-all -i ./000000-01-1.dcm -s Subject``

In this case, you just give Freesurfer the name of the very first dicom file in the directory, and it will find the rest of them.

**2. Using .nii or .mgz files:** you can type:
    ``recon-all -i /name/name/name/name/name.nii -s <subject_name>``

For example:
    ``/auto/myfolder/anatomy/Subject/Subject_t1_nii -s Subject``

The ‘-s Subject’ portion creates a folder, in this case a folder titled "Subject". The folder should be named for the subject.



Creating Pial & White Matter Surfaces
###########################################

Now that the anatomical data for your subject has been imported into Freesurfer and a new
directory has been created, you next want to run a command called autorecon1 to separate
the brain from the rest of the anatomy (eyes, muscle, etc.), and then manually check for and
correct potential errors. Next, you will need to determine the surfaces of the brain – both the
pial surface as well as the white matter. This is done using the command autorecon2.
Following this, you will manually make edits to these newly created surfaces.


Autorecon1: Separating Brain from Other Anatomy
***************************************************

The autorecon1 command motion corrects and conforms, normalizes, computes the Talairach transform, and strips the skull. Keep in mind that this command takes approximately 30-40 minutes, so make sure you're ready before running it. To use the command type:

    ``recon-all -autorecon1 -s <subject_name>``

For example:
    ``recon-all -autorecon1 -s Subject``


Manual Edits to the Anatomical
--------------------------------

At this stage, you just want to make sure that autorecon1 ran successfully and that large parts
of non-brain anatomy were not left behind. If big chunks of eye or skull were left behind, it is
good to manually delete them yourself. If autorecon1 ran successfully, you can probably skip
manual editing even if some anatomy was left behind since the next step, autorecon2, is quite
accurate at determining brain surfaces even if non-brain anatomy was left behind. However, it’s good to double check that everything worked out.

To pull up the newly stripped brains and make manual edits, type in your terminal:
    ``ipython``

    ``import cortex``

    ``cortex.segment.fix_wm(‘Subject’)``

This should cause three windows to pop up: a mayavi viewer with the 3D brain, one of the brain in 2D, and one of a tool bar. At this point, you want to edit individual voxels. This mostly consists of getting rid of remaining skull and eyes. To do this, click the edit voxels tool on the toolbox bar or press A on your keyboard as a shortcut. After this, to delete voxels, simply right click the areas you wish to delete. If you erase something by accident and want to undo it, press CTRL + Z (this only works for the last thing you erased so be careful).

In Mayavi:

- Left-click and drag to rotate.
- Middle-click and drag to pan.
- Right-click and drag to zoom.

Closing the mayavi window to automatically open the other hemisphere; close that to return to the first one. Left-click a point on the brain to save its location (a mark will be placed on the brain). This location can then be loaded in tkmedit. Look for red spikes and blue pockmarks on the brain - these usually indicate an incorrectly marked area on the white matter mask.
    

In tkmedit: 

*insert images here*

- Navigate - pan the image
- Edit Voxels Tool - your main tool when using tkmedit. 
- Left click to center the volume index at a given point. This is used to find the value of a voxel and to keep track of it when you change views.
- Center click to set a voxel value to 255 (default) or to clone to that voxel from the aux volume.
- Right click to clear a voxel.

- Main surface - the yellow curve used to generate the 3D model of the white matter surface.
- Original surface - the green curve, an unsmoothed version of the Main surface.
- Pial surface - the red curve marking the outer borders of the brain, the grey matter surface.
- Show Main Volume - the mask you are working on.
- Show Aux Volume - The full brain volume. 
- Coronal, Horizontal and Sagittal view - change the perspective you are viewing from. 

Reset view settings for zoom and offset.
Save or load a selected point for use with another program, such as the 3D models in mayavi. (Click this one to get to the point that you selected in the mayavi viewer.)

Again:
You can undo with ctrl+z, but it only remembers the last action done.
If you erase something by accident, or want to restore something:

Tools > Configure volume brush
    Set Mode to Clone
    Set Clone Source to Aux Volume

This lets you paint from the aux volume to the mask. 
Set Mode back to New Value if you’re done.

To change brush size:
Tools > Configure brush info > Change Radius

To change the size of the "paintbrush”, in the tool bar, go to: tools > configure brush info and
change the radius. A shortcut to do the same thing is to press the numbers on the keypad of
your keyboard (where 1 is 1x1, 4 is 4x4, etc).
Generally you should just work with a 1-pixel radius, though.
To save, just go to file > save in the tool bar.

|
    
When you are done:

File > Save Main Volume
File > Quit (the program may stumble a bit if you just close the window)
iPython will give you three options. 
1) Run autorecon-wm?
2) Run autorecon-pia?
3) Do nothing?
If you are finished with the mask, enter 1. Otherwise enter 3.

|



Autorecon2: Creating Surfaces
***********************************

Here, you will be creating both white and gray matter surfaces using the autorecon2
command. When the command is complete, there will be outlines on the brain indicating that
the program has determined where the pial and white matter surfaces are located. The pial
surface will be outlined in red, and the white matter surface will be outlined in both green and
yellow when it is finished running.

Type in the command:
    ``recon-all -autorecon2 -s <subject_name>``

*This will take up to 5 hours!*

|

Although at this stage Freesurfer has completed determining where the white matter and pial
surfaces are, it is not completely accurate, so next edits have to be made to correct these
mistakes. This is the most time-consuming part of the brain segmentation.

First fix big mistakes in the white matter surface. These include large swaths of gray matter
being identified as white matter when it shouldn't, and when big portions of white matter are
not labeled as white matter when they should be. The command to make these edits is the same as above:
    
    ``ipython``
    
    ``import cortex``
    
    ``cortex.segment.fix_wm(“subject”)``

We’ll look through the results of autorecon2, examining the white matter curve and masks, and then the pial (gray matter) curve. This can be a lengthy process; because it’s an entirely nonverbal task, I recommend listening to podcasts as you go.    

|

The yellow outline represents the smoothed white matter surfaces while the green outline is
the surface that most closely resembles the individual voxel edits you've made. The yellow
surface is the one that will be used for flat maps, however it is easier to use the green surface when making edits since it actually reflects the changes you made rather than the smoothed changes.
You want to make sure to delete voxels that the green and yellow surfaces encompass that
it shouldn't (such as gray matter and/or leftover pieces of eye or skull) as well as add voxels
(middle click) to regions that appear to have white matter but aren't included in the
green/yellow surfaces. Make sure to hit "A" to switch to edit mode.


Autorecon on the white matter surface should take about 2 hours. These manual edits are an iterative process; when it’s done, go back and look over the 3D surface, and make any changes that seem necessary. New spikes can appear in unexpected places, so three or four iterations may be needed, probably more if you are just starting to learn how to do it.







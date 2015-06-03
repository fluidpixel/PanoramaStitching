# PanoramaStitching
OSX Interface for 360° panorama stitching algorithm

This is a front end to the algorithm we use for stitching images together into 360° panoramas. The
panorams a cylindrical rather than spherical.

You will need add the latest framework for OpenCV from <http://opencv.org>. The iOS version appears to work well on OSX.
Place the framework in the PanoramaStitching folder.

The files for the algorithm itself are contained in the 'Stitching' folder:

BPRefiner - based on OpenCV algorithms, refines the camera parameter estimates based on feature matches.

general_stitching_algorithm - the algorithms which performs the stitching:
Image registration - feature finding, feature matching etc
Image processing - seam finding etc.
Compositing.

On iOS where the images are taken by the camera while the device is rotating, the feature finding part of this process runs
asyncronously while the device is waiting for another image.

KNOWN ISSUES:


TODO:

Add sample images
Add camera rotation information
Finish front-end to allow the user to choose input/output files.

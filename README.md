Separate trees
==============================

Overview
------------

Two appraoches for separating individual instances in case of overlapping segmentation. The approaches assume a symmetry in shape, as such can be useful for separating merged objects such as trees and cells. 


Approach 1a: Center based separation in raster space
------------------------------------
This approach first finds the object centers, and then relabels the image based upon weighted distance to these centers. So, if a given blob contains multiple centers, then it will be divide into various instances accordingly. Current implementation adds a large penalty for not-in-sight centers, and additionally performs majority filtering to remove the lone-pixels. 


Approach 1b: Erosion based separation in raster space
------------------------------------
In this approach, we erode the image at multiple levels and store that new instances that merge in the process. For example, if a given blob were two split into multiple instances during erosion, we will store them as new instances and continue the process. 

Approach 2: Separation in polygon space
------------------------------------
In this approach, we work on individual polygons instead of large rasters. At the core, this approach uses either 1a or 1b. The main advantage is that it can be optimized for the end task directly, e.g. if you are only interested in the center of the tree and it's area. 


<!-- GETTING STARTED -->
Getting Started
------------

1. Clone repo
   ```sh
   git clone git@github.com:ankitkariryaa/seperate_trees.git
   ```
2. Change your current working directory to the github repo
   ```sh
   cd seperate_trees
   ```   
3. Install dependencies 
   ```sh
   conda env create -f environment.yml
   ```
4. Activate environment 
   ```sh
   conda activate separate-trees
   ```
5. Run the scripts 
   ```sh
   python -m src.center_separation -h
   ```
   or
   ```
   python -m src.polygon_separation -i ./sample_vectorfiles -ft gpkg -p small --cpu 1 -oaa true -crd ./sample_images -crt tif
   ```
   
Samples
----------

Input image:
![image](https://user-images.githubusercontent.com/822583/142783316-b1cf7ddc-2583-4d92-a34d-1b63d175f9ec.png)


Predicted centers and pixel labels using center based separation technique (approach 1).
![image](https://user-images.githubusercontent.com/822583/142919674-2586f603-44c0-4509-ba0b-828de33fd65d.png)

In this image the centers were found in 108ms and the labels were found in 52 seconds on 1 core (single thread), and 24 seconds on 4 CPU cores. 

Predicted pixel labels using erosion based separation technique (approach 2).
![image](https://user-images.githubusercontent.com/822583/142921459-fc29918a-a1a3-4a5c-bb49-f58c6ed02156.png)
In this image the labels were found in 84.8 seconds on 4 CPU cores. The instances were eroded at [3,8,15] pixels from the boundary. A more granual erosion example [2,5,8,11,14,17] might lead to finer results but it will be slower.
   
Known issues
----------
1. Approach 1 (center based separation) can in somecases lead to lone pixels. It usually happens when the closest/expected center is not in sight.
2. Extremely slow processing with certain large images. I am still debugging this issue.

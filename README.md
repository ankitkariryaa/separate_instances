Separate trees
==============================

Overview
------------

Bunch of algorithms to be used during post-processing for separating individual trees. 


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
   python -m src.center_based_separation -h
   ```
   
Samples
----------

Input image:
![image](https://user-images.githubusercontent.com/822583/142783316-b1cf7ddc-2583-4d92-a34d-1b63d175f9ec.png)


Predicted centers and pixel labels:
![image](https://user-images.githubusercontent.com/822583/142783357-3a4b525e-b3f8-4c8c-b8c7-463243a91820.png)

In this image the centers are found in 108ms and the labels are found in 52 sec (single thread).

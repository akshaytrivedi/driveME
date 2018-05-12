# driveME

Tried writing my own MaskRCNN first(implementation in fair-maskrcnn). This is giving errors which are hard to debug and this implementation is very slow.

So moved to matterport Mask RCNN implementation

Using matterport for the competition  

1. Run matterport model on simple images (DONE)

2. Run matterport model on Kaggle dataset (DONE)

3. Evaluate the matterport model 
	Run on test set (Done)
	Submit test set predictions to Kaggle (Getting Errors)
	Spilt training data into train and validation set (In progress)
	Evaluate on validation set and visualize matterport model errors

4. How we can better same matterport model to improve prediction
	Training/Fintuning the model
	Landscapes dataset
	Test image augmentation (May be decreasing size)
	Ensemble of each category i.e we train one classifer for each category vs background
	Ensemble of each video i.e predicting for each image of a video
	Semantic Segmentation for Videos
	
5. Improving the model architecture



















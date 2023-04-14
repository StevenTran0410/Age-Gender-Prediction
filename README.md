# Age_Gender_Prediction
Personal project for learning and hobby 
This project using source @leondgarse and @serengil (Sorry if i forget some of other source :(( please contact me if you findout who i miss)

Model weights can be download from here: drive.google.com/drive/folders/12-JLEXMbxVvvXhhlj9tLo5q8QJixS99G?usp=sharing

Note: In this project the model was a little bit overfit. In order to make it less overfit i train the model again 3 time by using the same dataset (only shuffle them using split_train_test from scikit-learn) and the result was improve alot. And if you want to get better age prediction, please consider test cropping an image by your hand a few time in order to reconize a patter of which way of cropping face give the best result or you can just using the extract_face function from DeepFace library for faster (recommend using mediapipe as backbone for extracting). In the Drive there are 2 model, one is from ArcFace and the orther is FaceNet. This repo still under developement, upcoming is the GUI webcam app that use the model.

My contact: steven0410leminh@gmail.com

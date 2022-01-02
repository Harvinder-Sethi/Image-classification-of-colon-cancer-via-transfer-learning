# Image-classification-of-colon-cancer-via-transfer-learning
Design a reliable NN model to accurately determine whether a cell is benign(non-cancerous) or malignant(cancerous) and use the leaning of the developed model to further classify the cell types.

patch images link- https://drive.google.com/file/d/1gxBfEDukQGKD-AlHsTH4OwPnHCxTo4hr/view?usp=sharing

Introduction:
As a part of this assignment, we are trying to develop a neural network for classification of cells being cancerous and their type. We are using publicly available routine colon cancer histological dataset “CRCHistoPhenotypes”. The aim of this experiment is to design a reliable NN model to accurately determine whether a cell is benign(non-cancerous) or malignant(cancerous) and use the leaning of the developed model to further classify the cell types.

Performance Metric:
Since the dataset is imbalanced the performance metric we chose is ‘F1_score’ and ‘Accuracy’. We aim to develop a model with accuracy of at-least 70% and f1-score of at-least 0.7. The reason of selecting F1_score, rather than ‘Precision’ or ‘Recall’ is that we need minimize rate of both false positive and false negative prediction. If we focused on a model with a very high ‘Recall’, but because of its low ‘Precision’, we would have disproportionality predicted non-cancerous cells as cancerous, which is not ideal and vice-versa. Hence, we chose F1 score as our performance metric evaluator. To compare the performance between all the different model we have kept a constant ‘Epoch’ of 50 and ‘batch size’ of 32. The F1_score we are evaluating for each epoch is a weighted F1_score for the classification batch in each epoch. We chose to implement the weighted f1_score evaluation because of the imbalanced target class.As a loss function evaluator, we have selected ‘Binary Cross Entropy’ and ‘Sparse Cross Entropy’ for cancerous and cell type prediction, respectively. For the cancerous cell classification task, the target function had just two labels, therefore we need to evaluate the probability of a label belonging to either of the class. Hence, we implemented binary_cross_entropy as our loss function. For ‘cellType’ prediction we changed the loss function to ‘Sparse Categorical Cross Entropy’ as the target function has 4 different prediction labels. The reason we used Sparse Cross Entropy instead of Categorical Cross Entropy because our prediction labels are integers which are not one hot encoded.

Multi-Layer Perceptron Implementation:
The first model we explored was a MLP model. Compared to CNN these models are relatively simpler to implement, with comparatively less hyper tuning parameters. This makes design of a MLP model quite easy. The model we designed fully connected input layer of dimension (27x27x3), a single hidden layer of 256 connected filter and an output layer with 2 filters. For both the hidden layer and the output layer we used the default ‘sigmoid’ activation function as its value existing between 0 and 1. Hence, when we are trying to achieve binary classification, sigmoid activation is the perfect fit.

Hyper-Parameter Tuning:
With the base model implementation of MLP we got a model with a converging test and validation accuracy of around 86%, f1_score of 0.85 and oscillating validation loss of 0.35-0.40. The oscillation and non-convergence of test and validation loss in loss function graph hints towards the model which might not have achieved its global minima. To better the performance of model we implemented Ridge regularization (l2) as we need to avoid overfitting of model without losing any input features. ‘L2’ regularization adds ‘squared magnitude’ as penalty to the feature while ‘L1’ shrinks the feature. Since, we did not want to lose any feature while keeping the computational cost low we selected ‘L2’ regularization. For this model we selected regularization value of 0.001. We also changed the learning rate and momentum of the model, as it decides the step size at each iteration to find the minimum of loss function and help maintain the directing of step decent, respectively. We also implemented dropout technique to avoid overfitting as it randomly drops some of the connection from the previous layer, thereby reducing symmetry in the model. We settled on learning rate of 0.01 and momentum of 0.5. Ideally the value of ‘L2’ regularization, learning rate and momentum
should calculated using either grid search or random search method, but due to complexity of model and limited access to GPU computational units we implemented the trial and miss approach and selected these parameters.

Result:
Even though we were able to achieve the accuracy of around 84% , F1 score of 0.814 and a smooth loss function we can’t use this model as it doesn’t generalize well. The downside of this approach is that for a high dimensional dataset we need to perform manual feature selection, which can be a daunting task especially when there is lack of field knowledge. The complexity of feature selection exponentially increases with images, as most of the time we cannot make sense of the split images used for training, because of lack of domain knowledge. This problem is easily solved in CNN, as the feature identification and selection process are taken care by the convolutional networks, hence we opted to implement a CNN model.

CNN Implementation:
For CNN implementation we have developed a model with 4 convolution layers, input layer of dimension (27x27x3), connected layer of 128 filters and output layer with two filters. For the hidden layer we used ‘Relu’ activation while for the output layer we implemented ‘Softmax’ activation function. The reason we implemented instead of ‘sigmoid’ is because, ‘softmax’ the sum probability returned for the target labels is 1, hence the learning transfer is quite easy as we would be using the same, cancerous cell prediction model for cell type prediction. The reason we selected 4 convolution layer is to help model better learn and identify the features present in the input images. The convolution layers have padding set to ‘same’, adding zeros to boundaries of each input. This ensures the last convolution layer is not dropped if the input and pooling dimensions does not match.
We are using ‘Adam’ optimizer instead of ‘SGD’ because for most of the use case it finds the global minima faster than SGD. It tries to find the global minima by computing decaying average of past squared gradients and past gradient.

Hyper-Parameter Tuning:
The base model was highly over fitted with the training accuracy and F1_score reaching almost 100%. To resolve this issue like MLP model approach we implemented ‘L2’ regularization and dropout in the final layer. This led to model being not overfitted, but a substantial decrease in accuracy and f1_score. Note we did try dropout in every layer, but it did not provide any performance gain, hence we settled on dropout in just last layer.
To increase the size of the dataset we implemented data augmentation technique, which exposes the model to extended dataset made by flipping, rotating, or changing alpha values. We retrained the model on the augmented dataset with class weights assignment to each label. The reason we assigned custom class weight is to consider the class imbalance. The retrained model with data-augmentation, class weight assignment, ‘l2’ regularization and dropout in last layer was able to achieve classification accuracy of 85% with f1_score of 0.831 and virtually no overfitting. As mentioned in case of MLP, the ideal method to hyper tune the parameter is through either random search or grid search technique, but due to limited computational resources we opted for hit and try approach.

Transfer-Learning:
We use the previously trained CNN Model via Transfer learning to classify the different cell type. To achieve this, we loaded the previous CNN Model without the output layer, then we added a new o/p layer of dimension 4, corresponding to cell types which needs to be predicted. The new cell type prediction model had a decent accuracy of 74% but a low f1_score of 0.667. To improve the f1_score of the model we calculated the class weights of predicted labels based on their distribution using ‘sklearn’ ‘class_weight’ class. The model is built again with new class weight, resulting in ‘celltype’ prediction model with accuracy of 74% but an improved f1_score of 0.707. As a downside to adjusted class weight the model became a bit overfitted.

Independent Evaluation:
As we do not have access to histopathology images outside the dataset provided. Our evaluation is comparing our own work to 2 of the similar work/papers available online. paper1- locality Sensitive Deep Learning for Detection and Classification of Nuclei in Routine Colon Cancer Histology Images [1] and paper2- RCC Net: An Efficient Convolutional Neural Network for Histological Routine Colon Cancer Nuclei Classification [2].
Our Work Paper 1 Paper 2
      1) Tried MLP AND CNN (VGG) for ‘isCancerous’ prediction and used transfer learning from VGG to Classify ‘CellType’.
2) Dataset was provided in canvas and had downscaled images of 27*27*3 pixel with 20,280 image/patches.
3) Final structure for ‘isCancerous’ prediction is [0.1] and for ‘CellType’ prediction is [0.2]
4) In MLP optimizer=SGD, L2 Regularization used with activation= Sigmoid, Lr=0.01 and m=0.5, dropout=0.25 used. In VGG activation= SoftMax in o/p layer, class weights=balanced with epoch=50
5) Evaluated the models F1 and Accuracy using keras inbuilt function and sklearn confusion matrix.
6) MLP has achieved an accuracy of 84% and 0.83 weighted F1 score while VGG has achieved 80% and 0.78 respectively for ‘IsCancerous’. Transfer learning VGG has achieved 75% with weighted average F1 of 0.70.
   1) A spatially constrained Convolutional neural network used (SC-CNN) to locate the center of nuclei then a Novel Neighboring Ensemble Predictor (NEP) coupled with CNN is used to detect class label for that cell
2) Manual annotations of nuclei were conducted by experienced pathologist (YT) and a practitioner.
3) Dataset used is from TIA Centre Warwick [1] Uses 100 H&E-Stained images of 500*500*3 px with total 29,756 nuclei patches.
4) Precision-recall curve for nucleus detection. Isolines indicate regions of different F1 scores. The curve is generated by varying the value of threshold applied to the predicted probability map before locating local maxima and comparison result is plotted 
5) SRCNN gives the best precision of 0.783 AND SCCNN(M=1) gives the Recall of 0.827 WHILE best F1 Score= 0.802 is given by SCCNN(M=2)
   1) An efficient Convolutional Neural Network (CNN) based architecture for classification of histological routine colon cancer nuclei named as RCCNet is used.
2) Dataset used is publicly available on ‘CRCHistoPhenotypes’[1] and has 32*32- pixel image patches with total 22444 nuclei patches.
3) structure= 32*32*3 input -> 2 conv2D Layers-> pooling-> 2 conv2D Layers-> pooling->3 Fully connected (FC) layers 
4) Learning rate 6 × 10^(−5) then decreased iteratively, Relu activation applied, dropout=0.5 used after Relu of each FC Layer, batch normalisation used with epochs =500 and Adam’s Optimiser with decay=1*10^(-6)
5) The final proposed model has achieved a accuracy of 80.61% and 0.7887 weighted F1 score.
6) The results of RCCNet model is compared with 5 state-of-the-art CNN models like AlexNet, CIFAR-VGG, GoogLeNet, and WRN in terms of the accuracy, F1 score and training time 
 
References:

1. Sirinukunwattana K, Ahmed Raza SE, Yee-Wah Tsang, Snead DR, Cree IA, Rajpoot NM. Locality Sensitive Deep Learning for Detection and Classification of Nuclei in Routine Colon Cancer Histology Images. IEEE Trans Med Imaging. 2016 May;35(5):1196-1206. doi: 10.1109/TMI.2016.2525803. Epub 2016 Feb 4. PMID: 26863654.
2. S. H. Shabbeer Basha, S. Ghosh, K. Kishan Babu, S. Ram Dubey, V. Pulabaigari and S. Mukherjee, "RCCNet: An Efficient Convolutional Neural Network for Histological Routine Colon Cancer Nuclei Classification," 2018 15th International Conference on Control, Automation, Robotics and Vision (ICARCV), 2018, pp. 1222-1227, doi: 10.1109/ICARCV.2018.8581147.
3. 4.
2021]. 5.
6.
7.
Medium. 2021. From SGD to Adam. [online] Available at: <https://medium.com/mdr-inc/from-sgd-to-adam-
c9fce513c4bb> [Accessed 28 May 2021].
Medium. 2021. L1 and L2 Regularization Methods. [online] Available at:
<https://towardsdatascience.com/l1-and-l2-regularization-methods-ce25e7fc831c> [Accessed 28 May
other, C. and Rafałko, K., 2021. Cross Entropy vs. Sparse Cross Entropy: When to use one over the other.
[online] Cross Validated. Available at: <https://stats.stackexchange.com/questions/326065/cross-entropy-
vs-sparse-cross-entropy-when-to-use-one-over-the-other> [Accessed 30 May 2021].
Medium. 2021. Activation Functions in Neural Networks. [online] Available at:
<https://towardsdatascience.com/activation-functions-neural-networks-
1cbd9f8d91d6#:~:text=The%20main%20reason%20why%20we,the%20probability%20as%20an%20output.&
text=The%20logistic%20sigmoid%20function%20can,stuck%20at%20the%20training%20time.> [Accessed 30
May 2021].
Stanford.edu. 2021. CS 230 - Convolutional Neural Networks Cheatsheet. [online] Available at:
<https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks#style-
transfer> [Accessed 29 May 2021].

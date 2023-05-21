# Urban-Env-Detection-Proj1

The following model investigatio is done under time constraint due to issues with AWS.
Somehow I spent the 20 dollar limit after training and deploying my first model （EffDet) so hyperparameter tuning is skipped. Howevere, these will still be dicussed a high level basis.


## Issues
Unable to launch tensorboard properly.
Currently having the same issue as [post 985068 by Bernd](https://knowledge.udacity.com/questions/985068)
- Can only launch tensorboard using jupyter lab terminal
- No luck running it inside jupyter notebook :(
    - "No dashboards are active for the current data set."
- Even if I open the tensorboard page via terminal, I can only see the images in the "Images" section but the graphs in the other sections load forever.
- Unable to compare training vs. val accuracies.

## Overview
The models being experimented are the following:
- EfficientDet D1 640x640
- SSD MobileNet V1 FPN 640x640
- Faster R-CNN ResNet50 V1 640x640
- Due to resource and time constraints, all the `pipeline.config` file parameters are kept at defaul. 
- However, all three models are ran with a `batch_size = 8` and `num_epochs = 1000`
## Videos
- The video for Efficient Det is the one trained with 3000 epochs instead of 1000 as mentioned above. 
- This was trained using the provided 20 dollar budget that I somehow used up really fast ...
## Hypothesis
- Before training, some basic research have been done to get an overview of the three models.
### EfficientDet

- Its architecture is based on the EfficientNet backbone and uses a combination of feature pyramid networks and bi-directional feature pyramid networks (BiFPN) to capture multi-scale features.
- Generally "high accuracy while maintaining efficiency"        
### SSD MobileNet V1 FPN

- Single Shot MultiBox Detector
- Known for its real-time inference capabilities
- Its architecture uses a lightweight network designed for mobile and embedded devices, providing a good trade-off between speed and accuracy.
- FPN (Feature Pyramid Network) is incorporated to capture multi-scale features, aiding in detecting objects at different scales.
- Generally "fast but has relatively lower accuracies compared to other popular models"
### Faster R-CNN ResNet50

-  Two stages: 

    - a region proposal network (RPN) that generates potential object proposals
    - a classification network that classifies and refines these proposals.
- Uses ResNet50 as backbone, this architecture has50-layers and uses a bottleneck design for the building block
- Generally "more accurate but slower than single shot model"

### Discussion
Without running any tests and analyzing the report, I cannot say much regarding the accuracy of the models.
However, in terms of speed, Faster R-CNN ResNet50 seems to be slower than the other two due to its two stage nature.

In general, selecting a model depends on the specific requirements, such as the trade-off between accuracy and speed for specific application.
This project only wants us to compare the

## EfficientDet D1
### Hyperparameter Settings

### Test Result

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.081
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.207
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.052
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.027
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.334
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.380
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.022
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.091
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.133
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.075
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.426
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.472
```
 
 
## SSD MobileNet V2 FPNLite 640x640
 ### Hyperparameter Settings

### Test Result
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.077
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.163
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.064
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.028
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.291
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.385
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.022
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.082
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.119
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.061
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.412
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.423
```
 
 ## Discussion : EfficientDet vs. SSD MobileNet
 Note: Faster R-CNN ResNet50 V1 640x640 is still undeer training and it's taking way too much time to train (~20+ min per 100 steps). I am also currently using my own AWS resources to train the model. Therefore, the model might be discarded for comparison. The analysis will be done with the above two models for now.
 
### EfficientDet D1:

Average Precision (AP) ranges from 0.027 to 0.380 across different IoU thresholds and object sizes.
Average Recall (AR) ranges from 0.022 to 0.472 across different IoU thresholds and object sizes.

### SSD MobileNet V2 FPNLite:

Average Precision (AP) ranges from 0.028 to 0.385 across different IoU thresholds and object sizes.
Average Recall (AR) ranges from 0.022 to 0.423 across different IoU thresholds and object sizes.
Comparing the performance of the two models, both EfficientDet D1 and SSD MobileNet V2 FPNLite show similar trends in terms of AP and AR across different evaluation metrics. However, the magnitudes of the performance metrics differ slightly.

### Analysis
In terms of average precision, both models achieve similar AP values, with EfficientDet D1 having slightly higher AP scores for most object sizes. Similarly, in terms of average recall, EfficientDet D1 tends to have slightly higher AR values across different IoU thresholds and object sizes.

Considering these results, EfficientDet D1 appears to have a **slightly better** performance in terms of accuracy compared to SSD MobileNet V2 FPNLite. However, the differences between the models are not substantial, and other factors such as model speed and complexity should also be taken into account if we were to consider production level product.

## Improvement
### Learning Rate 
- The learning rate controls the step size in the optimization process. 
- One can try different learning rates to find the optimal value for the dataset.
- In addition, leanring rate schedulers can be used to adjust the learning rate over time. 
- For example, exponential decay **may** help improve convergence and performance by decreasing the learning rate as training progresses

### Batch Size
- Determines the number of samples processed in each training iteration.
-  A larger batch size can lead to faster convergence, but it also requires more memory. 
-  Smaller batch sizes can provide better generalization, but training might take longer (which is the case why my Faster R-CNN is still training right now...)
-  Experiment with different batch sizes to find the optimal value that balances training efficiency and accuracy can be done given enough time and resource.

### Number of Epochs
- Increasing the number of epochs allows the model to see more examples, potentially improving its ability to generalize. 
- However, too many epochs can lead to overfitting. 
- If I had more time/resource and Tensorboard didn't fail on me :( , I could monitor the model's performance on the validation set to determine the optimal number of epochs.

### Data Augmentation 
- Can increase the diversity of the training data, helping the model generalize better. 
- For example: random cropping, rotation, scaling, flipping, and color jittering. 
- Can improve the model's robustness.

### Model Architecture 
- TensorFlow's model zoo provides a variety of pre-trained models with different architectures and complexities.
- Can experiment with the models and find the ones that best suit this problem.

### Regularization 
- Weight decay or dropout can help prevent overfitting.
- Can help combat overfitting due to, for example, high model complexity and high number of epochs


### Others
- I've used ADAM optimization in the past and found it to be performing very well.
- Could be something to test in the future.

 


## Faster R-CNN ResNet50 V1 640x640
 ### Hyperparameter Settings


### Test Result

 
 ### Discussion
 
 

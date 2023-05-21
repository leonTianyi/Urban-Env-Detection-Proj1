# Urban-Env-Detection-Proj1

The following model investigatio is done under time constraint due to issues with AWS.
Somehow I spent the 20 dollar limit after training my first model so hyperparameter tuning is shortened/skipped.


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

## Hypothesis
- Before training, some basic research have been done to get an overview of the three models.
- EfficientDet

        - Its architecture is based on the EfficientNet backbone and uses a combination of feature pyramid networks and bi-directional feature pyramid networks (BiFPN) to capture multi-scale features.
        - Generally "high accuracy while maintaining efficiency"        
- SSD MobileNet V1 FPN

        - Single Shot MultiBox Detector
        - Known for its real-time inference capabilities
        - Its architecture uses a lightweight network designed for mobile and embedded devices, providing a good trade-off between speed and accuracy.
        - FPN (Feature Pyramid Network) is incorporated to capture multi-scale features, aiding in detecting objects at different scales.
        - Generally "fast but has relatively lower accuracies compared to other popular models"
- Faster R-CNN ResNet50

        -  Two stages: 
        
            - a region proposal network (RPN) that generates potential object proposals
            - a classification network that classifies and refines these proposals.
        - Uses ResNet50 as backbone, this architecture has50-layers and uses a bottleneck design for the building block
        - Generally "more accurate but slow"
        

Without running any tests and analyzing the report, I cannot say much regarding the accuracy of the models.
However, in terms of speed, Faster R-CNN ResNet50 seems to be slower than the other two due to its two stage nature.

In general, selecting a model depends on the specific requirements, such as the trade-off between accuracy and speed for specific application.
This project only wants us to compare the

## EfficientDet D1
### Hyperparameter Settings

- Hyperparameters:
    - epochs : 1000
    - `pipeline.config`: default
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
 ### Discussion
 
 
## SSD MobileNet V2 FPNLite 640x640
 ### Hyperparameter Settings

- Hyperparameters:
    - epochs : 1000
    - `pipeline.config`:
        - `batch_size = 8`
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
 
 ### Discussion
 

## Faster R-CNN ResNet50 V1 640x640
 ### Hyperparameter Settings

- Hyperparameters:
    - epochs : 1000 (cut to 800)
    - `pipeline.config`: default
### Test Result

 
 ### Discussion
 
 

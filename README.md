The source code is from https://github.com/shillyshallysxy/emotion_classifier


Since I am using a newer hardware platform than the original author's, the usage of many of the tool libraries has changed.
I reconfigured the platform, tweaked some of the code to run on my computer, and tuned and optimized the parameters of the model.
(source code model's acc is about 50%. After I adjusted the parameters, the accuracy was 66%)

Hadware: Intel(R) Core(TM) i7-10875H CPU 
                32.0 GB  RAM
                GTX2060
Software: Win10 pro 20h2
                python 3.8.3
                cuda 11.2.1
                cudnn 11.2
Library Requirements: keras 2.4.3
                                    OpenCV 4.5.1
                                    Sklearn
                                    
To run confusion_matrix.py, you need to add the folder name where the models are stored
eg. python confusion_matrix.py model

To run emotion_classifier.py, you need to add the folder name where the photo is stored
eg. python emotion_classifier.py 1
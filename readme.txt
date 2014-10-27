EvalLocalization ver1.0
2014/10/26 takuya minagawa

1. Overview
This program is C++ tool to evaluate object localization algorithms.
The way to evaluate is following Pascal VOC.

See:
Everingham, M., Gool, L., Williams, C. K. I., Winn, J., & Zisserman, A. (2009). The Pascal Visual Object Classes (VOC) Challenge. International Journal of Computer Vision, 88(2)


2. Install
You need boost and OpenCV to build this tool.
boost
http://www.boost.org/

OpenCV
http://opencv.org/

You can use pre-compiled version of windows. Extract EvalLocalization.zip and start "exe" file.
If it does not work, you may need to install VC++2013 runtime.

You can download it at:

http://www.microsoft.com/en-us/download/details.aspx?id=40784


3. How to Use
Command line format is:

EvalLocalization <localization file> <ground truth file> <output file> [option]

<localization file>
A text file that describes results of object localization.
The format is same as training file of OpenCV cascade classifier:
=================================
<image file path> <the number of object> <X of top left> <Y of top left> <width> <height> ....
  .
  .
  .
=================================

For instance, there are two objects detected such as (x,y,w,h)=(10,14,100,120), (141,151,100,120), then:
=====================================
folder/imagefile1.jpg 2 10 14 100 120 141 151 100 120
=====================================


<ground truth file>
A text file that describes correct label of objects in each image.
Its format is same as <localization file>.


<output file>
Output file that summarized evaluation results in CSV format.
====================================
<image file path>, <number of true positive>, <number of false positive>, <number of miss-detection>
  .
  .
  .
===================================


[option]
Here is the options of the command line:
-h                   Print help
-s <file path>       Indicate a file that describes a score of each detected object
-c <threshold>       Threshold of scores (default = 0.5)
-o <threshold>	     Threshold of overlap threshold (default = 0.5)
-d <directory path>  Directory path to save images in which localization result was drawn
-t <file path>	     Output text file of true positive
-f <file path>	     Output text file of false positive
-r <file path>	     Output CSV file of recall-precision curve


The below is an example of command line:
============================================
EvalLocalization.exe testResult.txt trueLocations.txt summary.csv -s testProb.txt -d ./Draw -t true_positive.txt -f false_positive.txt -r RP.csv -c 0.7 -o 0.5
============================================


4. Score file
You can indicate the scores (probability) of each detection with '-s' option.
The format is shown below:
========================================
<the number of objects> <score> <score> ...
========================================

The number of <score> is equal to <the number of objects>.
ex:
=============================================
2 0.864495 0.860051
2 0.913481 0.861791
1 0.901213
2 0.854541 0.755583
.
.
.
=============================================

Each line of this file is binded to the each line in <localization file>.
Each <score> is bineded to each object position in the same line of <localization file>:therefore, <the number of objects> must have the same value as the one in <localization file>.


5. Thresholds
There are two types of thresholds.
One is the threshold of score with '-c' option, and another is the threshold of overlap between two rectangles (detected and ground truth) with '-o' option.
Default values of both are 0.5.
(In Pascal VOC, the overlap threshld must be 0.5)


6. Output of True Positive and False Positive
With '-d' option, you can save images in which true positive and false positive are drawn.
The images are loaded from the first column of <localization file>, and true and flase positives are respectively drawn in blue and red.

The drawn images are saved as PNG format in the indicated directory.
File names are: 
1.png
2.png
.
.
.
Each number of filename is binded to the each line of <localization file>.

You can save these information of true and false positives in text files with '-t' and '-f' options.
Output format of these text files is the same as <localization file>.


This algorithm uses two thresholds (indicated with '-c' and '-o') to judge true positive and fale positive.



7. Recall-Precision Curve
You can output recall-precision curve into CSV file with '-r' option.
Too use this option, you must indicate '-s' option too.

Output format is as follow:
===============================
<threshold>, <recall>, <precision>
===============================
You can visualize recall-precision curve by creating scatter plot with recall and precision.

You can also get average precision in standard output.


8. License
This software is released under "MIT License".
http://opensource.org/licenses/MIT

Notice: The libraries used in this software (OpenCV and Boost) are followed under the license of each.


Takuya MINAGAWA (z.takmin@gmail.com)

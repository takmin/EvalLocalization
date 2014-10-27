EvalLocalization ver1.0
2014/10/26 takuya minagawa

1. �T�v
���̃v���O�����͕��̌��o�̌��ʂ��ʓI�ɕ]�����邽�߂̃v���O�����ł��B
�]�����@��Pascal VOC 2009�ɏ]���Ă��܂��B

Everingham, M., Gool, L., Williams, C. K. I., Winn, J., & Zisserman, A. (2009). The Pascal Visual Object Classes (VOC) Challenge. International Journal of Computer Vision, 88(2).


2. �C���X�g�[��
�r���h�ɂ́Aboost��OpenCV���K�v�ł��B
boost
http://www.boost.org/

OpenCV
http://opencv.org/

�R���p�C���ς݂̃o�[�W�������g�p����ꍇ�́AEvalLocalization.zip���𓀂���exe�t�@�C�������s���邾���ł��B
�������s�t�@�C�������܂������Ȃ��ꍇ�́AVC++2013�̃����^�C�����C���X�g�[������K�v�����邩������܂���B
�ȉ��̃T�C�g���炨�g���̃v���Z�b�T�ɂ����������^�C����T���A�_�E�����[�h�ƃC���X�g�[�������ĉ�����

http://www.microsoft.com/ja-jp/download/details.aspx?id=40784


3. �g����
�{�v���O�����̎g�����͈ȉ��̒ʂ�ł��B

EvalLocalization <localization file> <ground truth file> <output file> [option]

<localization file>
���o���ʂ��L�q�����t�@�C���ł��B
�ȉ��̃t�H�[�}�b�g�ɏ]���܂��B
=================================
�u�摜�t�@�C�����v �u���o���v �u����x���W�v �u����y���W�v �u���v �u�����v...
  .
  .
  .
=================================

�Ⴆ��(x,y,w,h)=(10,14,100,120), (141,151,100,120)�I�u�W�F�N�g���Q����P�[�X�̏ꍇ

=====================================
20100915-1/0000004.jpg 2 10 14 100 120 141 151 100 120
=====================================
�Ƃ����t�H�[�}�b�g�ɂȂ�܂��B


<ground truth file>
�������x���t�@�C���ł��B
<localization file>�Ɠ����t�H�[�}�b�g�ł��B


<output file>
�o�̓t�@�C���ł��B�ȉ��̂悤�ȃt�H�[�}�b�g��CSV�t�@�C���ł��B
====================================
�u�摜�t�@�C�����v,�u�������o���v,�u�댟�o���v,�u�����o���v
  .
  .
  .
===================================


[option]
�w��ł���I�v�V�����͈ȉ��̒ʂ�ł��B
-h	�w���v�̕\��
-s <file path>	�X�R�A�t�@�C���̎w��
-c <threshold>	�X�R�A��臒l�ݒ�i�f�t�H���g:0.5�j
-o <threshold>	�I�[�o�[���b�v��臒l��ݒ�i�f�t�H���g:0.5�j
-d <directory path>	���o���ʂ�`�悵���摜��ۑ�����t�H���_��
-t <file path>	���������o���ꂽ���ʂ̂ݏo�͂���e�L�X�g�t�@�C����
-f <file path>	�댟�o���ʂ̂ݏo�͂���e�L�X�g�t�@�C����
-r <file path>	recall-precision curve��CSV�`���ŏo��


�Ⴆ�΁A�ȉ��̂悤�Ȍ`�ŃR�}���h���g�p���܂��B
============================================
EvalLocalization.exe testResult.txt trueLocations.txt summary.csv -s testProb.txt -d ./Draw -t true_positive.txt -f false_positive.txt -r RP.csv -c 0.7 -o 0.5
============================================


4. �X�R�A�t�@�C��
'-s'�I�v�V�����Ŋe���o���ʂɕR�Â����X�R�A���w�肷�邱�Ƃ��ł��܂��B
�ȉ��̂悤�ȃt�H�[�}�b�g�ł��B
========================================
�u���o���v�@�u�X�R�A�v�@�u�X�R�A�v�@...
========================================

��L�̂悤�Ɂu�X�R�A�v�����o���̐��������ׂ܂��B
��:
=============================================
2 0.864495 0.860051
2 0.913481 0.861791
1 0.901213
2 0.854541 0.755583
.
.
.
=============================================

���̃t�@�C���̊e�s��<localization file>�̊e�s�ɑΉ����܂��B
�܂��A<localization file>�̊e���o���ʂɏ�L�́u�X�R�A�v���Ή����܂��B
���������āu���o���v�͏�L�̃X�R�A�t�@�C����<localization file>�ň�v���Ă���K�v������܂��B


5. 臒l�̎w��
�����ł͂Q��臒l���w�肷�邱�Ƃ��ł��܂��B
��͏�L�̃X�R�A�ɑΉ�����臒l��'-c'�I�v�V�����Ŏw�肵�܂��B
�������ground truth�ƌ��o���ʂ̋�`���m�̏d�Ȃ��臒l��'-o'�I�v�V�����Ŏw�肵�܂��B
�ǂ�����f�t�H���g��0.5�ł��B
���ɋ�`�̏d�Ȃ��臒l��Pascal VOC�ł�0.5�ƒ�߂��Ă��܂��B


6. ����/�񐳉��̏o��
'-d'�I�v�V�����Ŏw�肵���t�H���_���Ɍ��ʉ摜��ۑ����܂��B
���ʉ摜�́A<localization file>��1��ڂɋL�q�����t�@�C���p�X��ǂݎ��A���������o���ꂽ�ʒu��̋�`�A����Č��o���ꂽ�ʒu��Ԃ̋�`�ŕ`�悵�܂��B
�t�@�C������

1.png
2.png
.
.
.
�Ƃ������`��PNG�`���ŕۑ�����܂��B
�����Ŋe�t�@�C�����̔ԍ���<localization file>�̍s�ɑΉ����܂��B

�܂��A'-t'�����'-f'�I�v�V�����ŁA�����̏����e�L�X�g�t�@�C���֏o�͂��邱�Ƃ��ł��܂��B
�܂�'-t'�Ő������o����'-f'�Ō댟�o�������ꂼ��<localization file>�Ɠ��`���ŏo�͂��܂��B


���������o���댟�o���̔��f��'-c'��'-o'�I�v�V�����Ŏw�肵���Q��臒l�����ɍs���܂��B



7. Recall-Precision Curve
'-r'�I�v�V�������w�肷�邱�ƂŁARecall Precision Curve��CSV�`���ŏo�͂��邱�Ƃ��ł��܂��B
�������A���̃I�v�V�������w�肷��ɂ�'-s'�ŃX�R�A�t�@�C�����w�肵�Ă���K�v������܂��B
�o�̓t�H�[�}�b�g�͈ȉ��̒ʂ�ł��B
===============================
�u臒l�v�C�uRecall�v�C�uPrecision�v
===============================
Recall��Precision�ŎU�z�}���쐬�����RP�J�[�u�������ł��܂��B

�܂��W���o�͂���average precision���o�͂��܂��B


8. ���C�Z���X
�{�\�t�g�E�F�A��"MIT License"�Ō��J���܂��B
MIT���C�Z���X�ɂ��ẮA�������URL�Q�ƁB

http://opensource.org/licenses/MIT

�������A�{���C�u�����̊J���ɗp���Ă���OpenCV�����Boost�Ɋւ��ẮA���ꂼ��̃��C�u�����̃��C�Z���X�ɏ����܂��B


�F����(z.takmin@gmail.com)

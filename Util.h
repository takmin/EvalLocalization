/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//
// Copyright (C) 2014 Takuya MINAGAWA.
// Third party copyrights are property of their respective owners.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
// of the Software, and to permit persons to whom the Software is furnished to do
// so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
// PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//
//M*/

#ifndef __UTIL__
#define __UTIL__

#include <opencv2/core/core.hpp>

namespace util{

	//! �A�m�e�[�V�����t�@�C���̓ǂݍ���
	/*!
	opencv_createsamles.exe�Ɠ��`���̃A�m�e�[�V�����t�@�C���ǂݏ���
	ReadCsvFile()�֐��K�{
	\param[in] gt_file �A�m�e�[�V�����t�@�C����
	\param[out] imgpathlist �摜�t�@�C���ւ̃p�X
	\param[out] rectlist �e�摜�ɂ���ꂽ�A�m�e�[�V�����̃��X�g
	\return �ǂݍ��݂̐���
	*/
	bool LoadAnnotationFile(const std::string& gt_file, std::vector<std::string>& imgpathlist, std::vector<std::vector<cv::Rect>>& rectlist);

	//! �A�m�e�[�V�����t�@�C���̕ۑ�
	/*!
	opencv_createsamles.exe�Ɠ��`���̃A�m�e�[�V�����t�@�C���ǂݏ���
	\param[in] anno_file �A�m�e�[�V�����t�@�C����
	\param[in] img_files �摜�t�@�C���ւ̃p�X
	\param[in] obj_rects �e�摜�ɂ���ꂽ�A�m�e�[�V�����̃��X�g
	\return �ۑ��̐���
	*/
	bool SaveAnnotationFile(const std::string& anno_file, const std::vector<std::string>& img_files, const std::vector<std::vector<cv::Rect>>& obj_rects, const std::string& sep = " ");

	//! �X�R�A�t�@�C���̓ǂݍ���
	bool LoadScoreFile(const std::string& score_file, std::vector<std::vector<float>>& scores);


	bool ReadCSVFile(const std::string& input_file, std::vector<std::vector<std::string>>& output_strings,
		const std::vector<std::string>& separater_vec = std::vector<std::string>());;

	std::vector<std::string> TokenizeString(const std::string& input_string, const std::vector<std::string>& separater_vec);

	void DrawTrueAndFalsePositive(const cv::Mat& img, cv::Mat& dst_img,
		const std::vector<cv::Rect>& true_positives, const std::vector<cv::Rect>& false_positives,
		int thickness = 1);

	template <typename T>
	int CountVectorElements(const std::vector<std::vector<T>>& vec){
		int total = 0;
		std::vector<std::vector<T>>::const_iterator it = vec.begin(), it_e = vec.end();
		while (it != it_e){
			total += it->size();
			it++;
		}
		return total;
	};
}

#endif
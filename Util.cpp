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

#include "Util.h"
#include <fstream>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

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
	bool LoadAnnotationFile(const std::string& gt_file, std::vector<std::string>& imgpathlist, std::vector<std::vector<cv::Rect>>& rectlist)
	{
		std::vector<std::vector<std::string>> tokenized_strings;
		std::vector<std::string> sep;
		sep.push_back(" ");
		if (!ReadCSVFile(gt_file, tokenized_strings, sep))
			return false;

		std::vector<std::vector<std::string>>::iterator it, it_end = tokenized_strings.end();
		for (it = tokenized_strings.begin(); it != it_end; it++){
			int num_str = it->size();
			if (num_str < 2)
				continue;

			std::string filename = (*it)[0];
			if (filename.empty() || filename.find("#") != std::string::npos){
				continue;
			}

			imgpathlist.push_back(filename);
			int obj_num = atoi((*it)[1].c_str());
			std::vector<cv::Rect> rects;
			for (int i = 0; i<obj_num && 4 * i + 6 <= num_str; i++){
				int j = 4 * i + 2;
				cv::Rect obj_rect;
				obj_rect.x = atoi((*it)[j].c_str());
				obj_rect.y = atoi((*it)[j + 1].c_str());
				obj_rect.width = atoi((*it)[j + 2].c_str());
				obj_rect.height = atoi((*it)[j + 3].c_str());
				rects.push_back(obj_rect);
			}
			rectlist.push_back(rects);
		}

		return true;
	}


	//! �A�m�e�[�V�����t�@�C���̕ۑ�
	/*!
	opencv_createsamles.exe�Ɠ��`���̃A�m�e�[�V�����t�@�C���ǂݏ���
	\param[in] anno_file �A�m�e�[�V�����t�@�C����
	\param[in] img_files �摜�t�@�C���ւ̃p�X
	\param[in] obj_rects �e�摜�ɂ���ꂽ�A�m�e�[�V�����̃��X�g
	\return �ۑ��̐���
	*/
	bool SaveAnnotationFile(const std::string& anno_file, const std::vector<std::string>& img_files, const std::vector<std::vector<cv::Rect>>& obj_rects, const std::string& sep)
	{
		assert(img_files.size() == obj_rects.size());

		std::ofstream ofs(anno_file);
		if (!ofs.is_open())
			return false;

		int num = img_files.size();
		for (int i = 0; i<num; i++){
			ofs << img_files[i] << sep << obj_rects[i].size();
			for (int j = 0; j<obj_rects[i].size(); j++){
				cv::Rect rect = obj_rects[i][j];
				ofs << sep << rect.x << sep << rect.y << sep << rect.width << sep << rect.height;
			}
			ofs << std::endl;
		}

		return true;
	}


	//! �X�R�A�t�@�C���̓ǂݍ���
	bool LoadScoreFile(const std::string& score_file, std::vector<std::vector<float>>& scores)
	{
		std::vector<std::vector<std::string>> tokenized_strings;
		std::vector<std::string> sep;
		sep.push_back(" ");
		if (!ReadCSVFile(score_file, tokenized_strings, sep))
			return false;

		std::vector<std::vector<std::string>>::iterator it, it_end = tokenized_strings.end();
		for (it = tokenized_strings.begin(); it != it_end; it++){
			int num_str = it->size();
			if (num_str < 1)
				continue;

			std::string first_char = (*it)[0];
			if (first_char.empty() || first_char.find("#") != std::string::npos){
				continue;
			}

			std::vector<float> values;
			int obj_num = atoi((*it)[0].c_str());
			if (obj_num > it->size() - 1){
				std::cerr << "Error: illegal format at line " << std::distance(tokenized_strings.begin(), it) + 1 
					<< " in " << score_file << std::endl;
				return false;
			}
			for (int i = 0; i<obj_num; i++){
				int j = i + 1;
				values.push_back(atof((*it)[j].c_str()));
			}
			scores.push_back(values);
		}

		return true;
	}


	bool ReadCSVFile(const std::string& input_file, std::vector<std::vector<std::string>>& output_strings,
		const std::vector<std::string>& separater_vec)
	{
		std::vector<std::string> sep_vec;
		if (separater_vec.empty()){
			sep_vec.push_back(",");
		}
		else{
			sep_vec = separater_vec;
		}
		std::ifstream ifs(input_file);
		if (!ifs.is_open())
			return false;

		output_strings.clear();

		std::string buf;
		while (ifs && std::getline(ifs, buf)){
			std::vector<std::string> str_list = TokenizeString(buf, sep_vec);
			output_strings.push_back(str_list);
		}
		return true;
	}


	std::vector<std::string> TokenizeString(const std::string& input_string, const std::vector<std::string>& separater_vec)
	{
		std::vector<std::string>::const_iterator separater_itr;
		std::vector<std::string::size_type>	index_vec;
		std::string::size_type	index;
		for (separater_itr = separater_vec.begin(); separater_itr != separater_vec.end(); separater_itr++){
			index = 0;
			while (true){
				index = input_string.find(*separater_itr, index);
				if (index == std::string::npos){
					break;
				}
				else{
					index_vec.push_back(index);
					index++;
				}
			}
		}
		sort(index_vec.begin(), index_vec.end());

		std::vector<std::string> ret_substr_vec;
		std::vector<std::string::size_type>::iterator idx_itr;
		std::string::size_type start_idx = 0;
		int str_size;
		for (idx_itr = index_vec.begin(); idx_itr != index_vec.end(); idx_itr++){
			str_size = *idx_itr - start_idx;
			ret_substr_vec.push_back(input_string.substr(start_idx, str_size));
			start_idx = *idx_itr + 1;
		}
		ret_substr_vec.push_back(input_string.substr(start_idx));

		return ret_substr_vec;
	}


	void DrawTrueAndFalsePositive(const cv::Mat& img, cv::Mat& dst_img,
		const std::vector<cv::Rect>& true_positives, const std::vector<cv::Rect>& false_positives,
		int thickness)
	{
		if (img.channels() == 1){
			cv::cvtColor(img, dst_img, cv::COLOR_GRAY2BGR);
		}
		else if(img.channels() == 3){
			dst_img = img;
		}
		else return;

		std::vector<cv::Rect>::const_iterator it;
		for (it = false_positives.begin(); it != false_positives.end(); it++){
			cv::rectangle(dst_img, *it, cv::Scalar(0, 0, 255), thickness);
		}
		for (it = true_positives.begin(); it != true_positives.end(); it++){
			cv::rectangle(dst_img, *it, cv::Scalar(255, 0, 0), thickness);
		}
	}

}
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

/******************************************************************************
EvalLocalization <localization file> <ground truth file> <output file name>
	-s <score file> -t <true positive> -f <false positive> -d <result directory>

Output:
Recall-Precision Curve with '-s' option
Recall-Precision Curve of area
*******************************************************************************/

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <fstream>
#include <iostream>
#include "Util.h"
#include "EvalFunctions.h"

#ifdef _DEBUG
#include <gtest/gtest.h>
#endif

using namespace boost::program_options;

void print_help(int argc, char * argv[], const options_description& opt)
{
	std::cout << argv[0] << " <localization file> <ground truth file> <output file> [option]" << std::endl;
	std::cout << opt << std::endl;
}


bool ParseCommandLine(int argc, char * argv[],
	std::string& localization_file, std::string& ground_truth, std::string& output_file,
	std::string& score_file, float* threshold, float* overlap_th,
	std::string& rp_file, std::string& draw_directory, 
	std::string& true_positive, std::string& false_positive)
{
	// option argments
	options_description opt("option");
	opt.add_options()
		("help,h", "Print help")
		("score,s", value<std::string>(), "score file name binded to input localization file")
		("scoreTh,c", value<float>()->default_value(0.5), "threshold of score")
		("overlapTh,o", value<float>()->default_value(0.5), "threshold of overlap")
		("draw,d", value<std::string>(), "directory to save result images which draw true positive and false positive with threshold '-st' and '-ot'")
		("truePos,t", value<std::string>(), "output true positive file with threshold '-s' and '-o'")
		("falsePos,f", value<std::string>(), "output false positive file with threshold '-s' and '-o'")
		("rpCurve,r", value<std::string>(), "generate recall-precision curve with threshold '-o'");

	variables_map argmap;
	try{
		// get command line argments
		store(parse_command_line(argc, argv, opt), argmap);
		notify(argmap);

		// print help
		if (argmap.count("help") || argc < 4){
			print_help(argc, argv, opt);
			return false;
		}

		localization_file = argv[1];
		ground_truth = argv[2];
		output_file = argv[3];
		if (localization_file.find("-") == 0 || ground_truth.find("-") == 0 || output_file.find("-") == 0){
			print_help(argc, argv, opt);
			return false;
		}

		*threshold = argmap["scoreTh"].as<float>();
		*overlap_th = argmap["overlapTh"].as<float>();

		if (!argmap["score"].empty())
			score_file = argmap["score"].as<std::string>();
		if (!argmap["draw"].empty())
			draw_directory = argmap["draw"].as<std::string>();
		if (!argmap["truePos"].empty())
			true_positive = argmap["truePos"].as<std::string>();
		if (!argmap["falsePos"].empty())
			false_positive = argmap["falsePos"].as<std::string>();
		if (!argmap["rpCurve"].empty())
			rp_file = argmap["rpCurve"].as<std::string>();
	}
	catch (std::exception& e)
	{
		std::cout << std::endl << e.what() << std::endl;
		print_help(argc, argv, opt);
		return false;
	}

	return true;
}


bool SaveSummary(const std::string& output_file,
	const std::vector<std::string>& img_files,
	const std::vector<std::vector<cv::Rect>>& ground_truth,
	const std::vector<std::vector<cv::Rect>>& true_positives,
	const std::vector<std::vector<cv::Rect>>& false_positives)
{
	assert(img_files.size() == ground_truth.size());
	assert(img_files.size() == true_positives.size());
	assert(false_positives.size() == true_positives.size());

	std::ofstream ofs(output_file);
	if (!ofs.is_open())
		return false;

	ofs << "file name,true positive,false positive,miss detect" << std::endl;
	int num_plot = img_files.size();
	for (int i = 0; i < num_plot; i++){
		ofs << img_files[i] << "," << true_positives[i].size() << "," 
			<< false_positives[i].size() << ","
			<< ground_truth[i].size() - true_positives[i].size() << std::endl;
	}
	std::cout << "Save summary file in " << output_file << " as CSV." << std::endl;

	return true;
}


bool SaveRecallPrecisionCurve(const std::string& output_file, 
	const std::vector<float>& recall, const std::vector<float>& precision, const std::vector<float>& thresholds)
{
	assert(recall.size() == precision.size());
	assert(recall.size() == thresholds.size());

	std::ofstream ofs(output_file);
	if (!ofs.is_open())
		return false;

	ofs << "threshold,recall,precision" << std::endl;
	int num_plot = recall.size();
	for (int i = 0; i < num_plot; i++){
		ofs << thresholds[i] << "," << recall[i] << "," << precision[i] << std::endl;
	}
	std::cout << "Save rp-curve in " << output_file << " as CSV: in order threshold, recall, and precision." << std::endl;

	return true;
}


bool DrawTrueAndFalsePositives(const std::vector<std::string>& filenames, const std::string& output_folder,
	const std::vector<std::vector<cv::Rect>>& true_positives, const std::vector<std::vector<cv::Rect>>& false_positives,
	int thickness)
{
	assert(filenames.size() == true_positives.size());
	assert(filenames.size() == false_positives.size());

	using namespace boost::filesystem;

	path dir_path(output_folder);
	if (!is_directory(dir_path)){
		std::cerr << "Error: " << output_folder << " is not a directory.";
		return false;
	}

	int N = filenames.size();
	for (int i = 0; i < N; i++){
		cv::Mat img = cv::imread(filenames[i]);
		std::cout << "Load " << filenames[i] << "...";
		if (img.empty()){
			std::cerr << "Error: Fail to load " << filenames[i] << std::endl;
			continue;
		}

		cv::Mat draw_img;
		util::DrawTrueAndFalsePositive(img, draw_img, true_positives[i], false_positives[i], thickness);

		std::stringstream str;
		str << i + 1 << ".png";
		path save_path = dir_path / path(str.str());
		std::string save_name = save_path.generic_string();
		bool write_ret = cv::imwrite(save_name, draw_img);
		if (!write_ret){
			std::cerr << "Error: Fail to save " << save_name << std::endl;
			continue;
		}
		std::cout << "Save as " << save_name << std::endl;
	}
	return true;
}


int main(int argc, char * argv[])
{
#ifdef _DEBUG
	::testing::InitGoogleTest(&argc, argv);

	std::cout << "Test finished with code " << RUN_ALL_TESTS() << std::endl;
#endif

	std::string local_file, ground_truth, output_file, rp_file,
		score_file, true_pos_file, false_pos_file, output_dir;
	float thresh, overlap_th;
	if (!ParseCommandLine(argc, argv, local_file, ground_truth, output_file,
		score_file, &thresh, &overlap_th, rp_file, output_dir, true_pos_file, false_pos_file))
		return -1;

	std::vector<std::string> img_files;
	std::vector<std::vector<cv::Rect>> positions;
	if (!util::LoadAnnotationFile(local_file, img_files, positions)){
		std::cerr << "Fail to load " << local_file << std::endl;
		return -1;
	}

	std::vector<std::string> gt_img_files;
	std::vector<std::vector<cv::Rect>> gt_positions;
	if (!util::LoadAnnotationFile(ground_truth, gt_img_files, gt_positions)){
		std::cerr << "Fail to load " << ground_truth << std::endl;
		return -1;
	}

	std::vector<std::vector<float>> scores;
	if (!score_file.empty()){
		if (!util::LoadScoreFile(score_file, scores)){
			std::cerr << "Fail to load " << score_file << std::endl;
			return -1;
		}
	}
	
	std::vector<float> recall, precision, thresholds;
	std::vector<std::vector<cv::Rect>> true_positives, false_positives;
	float ap = 0;
	if (scores.empty()){
		eval::EvaluateAll(positions, gt_positions, true_positives, false_positives, overlap_th);
	}
	else{
		eval::EvaluateAll(positions, gt_positions, scores, thresh, recall, precision, thresholds, true_positives, false_positives, &ap, overlap_th);
	}

	if (!SaveSummary(output_file, img_files, gt_positions, true_positives, false_positives)){
		std::cerr << "Error: Fail to save summary file " << output_file << std::endl;
	}

	if (!true_pos_file.empty()){
		// save result annotation file
		if (!util::SaveAnnotationFile(true_pos_file, img_files, true_positives))
			std::cerr << "Error: Fail to save true positive file " << true_pos_file << std::endl;;
	}

	if (!false_pos_file.empty()){
		// save result annotation file
		if (!util::SaveAnnotationFile(false_pos_file, img_files, false_positives))
			std::cerr << "Error: Fail to save false positive file " << false_pos_file << std::endl;
	}

	if (!output_dir.empty()){
		// draw result on input images and save
		DrawTrueAndFalsePositives(img_files, output_dir, true_positives, false_positives, 3);
	}

	// save recall precision
	if (!scores.empty() && !rp_file.empty()){
		if (!SaveRecallPrecisionCurve(rp_file, recall, precision, thresholds)){
			std::cerr << "Error: Fail to save rp-curve in " << rp_file << std::endl;
		}
		std::cout << "Average Precision: " << ap;
	}

	return 0;
}



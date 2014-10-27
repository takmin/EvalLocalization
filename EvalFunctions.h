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

#ifndef __EVAL_FUNCTIONS__
#define __EVAL_FUNCTIONS__

#include <opencv2/core/core.hpp>

namespace eval{

	// ハンガリー法
	void HangarianAlgorithm(const cv::Mat_<float>& cost, std::vector<int>& permutation);

	// 検出オブジェクトとGround Truthデータを結びつける
	void bindRectPairs(const std::vector<std::vector<cv::Rect>>& detect_positions,
		const std::vector<std::vector<cv::Rect>>& ground_truth,
		const std::vector<std::vector<float>>& scores,
		std::vector<std::vector<int>>& binded_index,
		std::vector<std::vector<float>>& overlap_score,
		int* ground_truth_num, float overlap_threshold = 0.5);


	inline void bindRectPairs(const std::vector<std::vector<cv::Rect>>& detect_positions,
		const std::vector<std::vector<cv::Rect>>& ground_truth,
		std::vector<std::vector<int>>& binded_index,
		std::vector<std::vector<float>>& overlap_score,
		int* ground_truth_num){
		bindRectPairs(detect_positions, ground_truth, std::vector<std::vector<float>>(), binded_index, overlap_score, ground_truth_num);
	};

	/*
	void bindRectPairs(const std::vector<std::vector<cv::Rect>>& detect_positions,
		const std::vector<std::vector<cv::Rect>>& ground_truth,
		std::vector<std::vector<int>>& binded_index,
		std::vector<std::vector<float>>& overlap_score,
		int* ground_truth_num);
		*/
	float RecallPrecision(const std::vector<std::vector<int>>& binded_idx,
		const std::vector<std::vector<float>>& scores,
		const std::vector<std::vector<float>>& overlap_score,
		int ground_truth_num,
		std::vector<float>& recall, std::vector<float>& precision, std::vector<float>& thresholds,
		float overlap_threshold = 0.5);

	inline float RecallPrecision(const std::vector<std::vector<int>>& binded_idx,
		const std::vector<std::vector<float>>& overlap_score,
		int ground_truth_num,
		std::vector<float>& recall, std::vector<float>& precision, std::vector<float>& thresholds,
		float overlap_threshold = 0.5){
		return RecallPrecision(binded_idx, std::vector<std::vector<float>>(), overlap_score, ground_truth_num,
			recall, precision, thresholds, overlap_threshold);
	};

	void EvaluateDetection(const std::vector<std::vector<int>>& binded_index,
		const std::vector<std::vector<float>>& scores,
		const std::vector<std::vector<float>>& overlap_scores,
		std::vector<std::vector<int>>& true_positive_id,
		std::vector<std::vector<int>>& false_positive_id,
		float threshold,
		float overlap_threshold = 0.5);

	inline void EvaluateDetection(const std::vector<std::vector<int>>& binded_index,
		const std::vector<std::vector<float>>& overlap_scores,
		std::vector<std::vector<int>>& true_positive_id, std::vector<std::vector<int>>& false_positive_id, 
		float overlap_threshold = 0.5)
	{
		EvaluateDetection(binded_index, std::vector<std::vector<float>>(), overlap_scores,
			true_positive_id, false_positive_id, 0, overlap_threshold);
	};


	void EvaluateAll(const std::vector<std::vector<cv::Rect>>& detect_positions,
		const std::vector<std::vector<cv::Rect>>& ground_truth,
		std::vector<std::vector<cv::Rect>> &true_positives, 
		std::vector<std::vector<cv::Rect>> &false_positives,
		float overlap_th = 0.5);


	void EvaluateAll(const std::vector<std::vector<cv::Rect>>& detect_positions,
		const std::vector<std::vector<cv::Rect>>& ground_truth,
		const std::vector<std::vector<float>>& scores,
		float threshold,
		std::vector<float>& recall, std::vector<float>& precision, std::vector<float>& thresholds,
		std::vector<std::vector<cv::Rect>> &true_positives, std::vector<std::vector<cv::Rect>> &false_positives,
		float* average_precision,
		float overlap_th = 0.5);


	void ThresholdDetectPositions(const std::vector<std::vector<cv::Rect>>& detect_positions,
		const std::vector<std::vector<float>>& scores, float threshold,
		std::vector<std::vector<cv::Rect>>& output_positions);


	void Id2Positions(const std::vector<std::vector<cv::Rect>>& all_positions, 
		const std::vector<std::vector<int>>& position_id, std::vector<std::vector<cv::Rect>>& positions);

}

#endif
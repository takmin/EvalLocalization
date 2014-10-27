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

#include "EvalFunctions.h"
#include "argsort.hpp"
#include "Util.h"

using namespace std;

namespace eval{


	//! ２つの矩形のオーバーラップの比率を算出。計算方法はPASCAL VOC準拠（0-1）
	float calcRectOverlap(const cv::Rect& InputRect, const cv::Rect& CorrectRect)
	{
		int l1 = InputRect.x;
		int r1 = l1 + InputRect.width;
		int t1 = InputRect.y;
		int b1 = t1 + InputRect.height;

		int l2 = CorrectRect.x;
		int r2 = l2 + CorrectRect.width;
		int t2 = CorrectRect.y;
		int b2 = t2 + CorrectRect.height;

		int l3, r3, t3, b3;
		if (l2 <= l1 && l1 < r2){
			l3 = l1;
		}
		else if (l1 <= l2 && l2 < r1){
			l3 = l2;
		}
		else{
			return 0;
		}

		if (t2 <= t1 && t1 < b2){
			t3 = t1;
		}
		else if (t1 <= t2 && t2 < b1){
			t3 = t2;
		}
		else{
			return 0;
		}

		if (r1 < r2){
			r3 = r1;
		}
		else{
			r3 = r2;
		}

		if (b1 < b2){
			b3 = b1;
		}
		else{
			b3 = b2;
		}

		float overlap_area = (r3 - l3)*(b3 - t3);
		float summed_area = InputRect.width * InputRect.height + CorrectRect.width * CorrectRect.height;

		return overlap_area / (summed_area - overlap_area);
	}


	void HangarianAlgorithm(const cv::Mat_<float>& cost, std::vector<int>& permutation)
	{
		if (cost.cols == 1 && cost.rows == 1){
			permutation.push_back(0);
			return;
		}

		cv::Mat tmp_cost = cost;
		double minVal, maxVal;
		if (cost.cols > 1)
		for (int r = 0; r < cost.rows; r++){
			cv::minMaxIdx(tmp_cost(cv::Rect(0, r, cost.cols, 1)), &minVal, &maxVal);
			tmp_cost(cv::Rect(0, r, cost.cols, 1)) -= minVal;
		}

		if (cost.rows > 1)
		for (int c = 0; c < cost.cols; c++){
			cv::minMaxIdx(tmp_cost(cv::Rect(c, 0, 1, cost.rows)), &minVal, &maxVal);
			tmp_cost(cv::Rect(c, 0, 1, cost.rows)) -= minVal;
		}

		// search zero idx
		cv::Mat zero_map = cv::Mat::zeros(tmp_cost.size(), CV_32SC1);
		for (int r = 0; r < cost.rows; r++){
			for (int c = 0; c < cost.cols; c++){
				if (tmp_cost.at<float>(r, c) == 0){
					zero_map.at<int>(r,c) = 1;
				}
			}
		}

		bool exit_flag = false;
		std::vector<cv::Point> bind_pts;
		while (!exit_flag){
			exit_flag = true;
			for (int r = 0; r < cost.rows; r++){
				cv::Point pt;
				int col_count = 0;
				for (int c = 0; c < cost.cols; c++){
					if (zero_map.at<int>(r, c) > 0){
						col_count++;
						if (col_count > 1)
							break;
						pt.x = c, pt.y = r;
					}
				}
				if (col_count == 1){
					bind_pts.push_back(pt);
					zero_map(cv::Rect(pt.x, 0, 1, zero_map.rows)) = 0;
					zero_map(cv::Rect(0, pt.y, zero_map.cols, 1)) = 0;
					exit_flag = false;
					break;
				}
			}
		}

		permutation.resize(cost.cols, -1);
		for (int i = 0; i < bind_pts.size(); i++){
			permutation[bind_pts[i].x] = bind_pts[i].y;
		}
	}



	// 検出オブジェクトとGround Truthデータを結びつける
	void bindRectPairs(const std::vector<cv::Rect>& detect_positions, const std::vector<cv::Rect>& ground_truth,
		std::vector<int>& binded_index, std::vector<float>& overlap_score)
	{
		int w = detect_positions.size();
		int h = ground_truth.size();
		if (h < w)
			h = w;

		cv::Mat score_matrix(h, w, CV_32FC1);
		for (int r = 0; r < ground_truth.size(); r++){
			for (int c = 0; c < w; c++){
				score_matrix.at<float>(r, c) = calcRectOverlap(detect_positions[c], ground_truth[r]);
			}
		}
		for (int r = ground_truth.size(); r < h; r++){
			for (int c = 0; c < w; c++){
				score_matrix.at<float>(r, c) = 0;
			}
		}
		cv::Mat cost_matrix = -score_matrix + 1;

		std::vector<int> permutation;
		HangarianAlgorithm(cost_matrix, permutation);

		binded_index.resize(detect_positions.size());
		overlap_score.resize(detect_positions.size(), 0);
		for (int i = 0; i < permutation.size(); i++){
			binded_index[i] = permutation[i];
			if (permutation[i] >= 0)
				overlap_score[i] = score_matrix.at<float>(permutation[i], i);
			if (permutation[i] >= ground_truth.size())
				binded_index[i] = -1;
		}
	}
	


	// 検出オブジェクトとGround Truthデータを結びつける
	void bindRectPairs(const std::vector<cv::Rect>& detect_positions, const std::vector<cv::Rect>& ground_truth, const std::vector<float>& scores,
		std::vector<int>& binded_index, std::vector<float>& overlap_score, float overlap_threshold)
	{
		assert(scores.empty() || detect_positions.size() == scores.size());

		if (scores.empty()){
			bindRectPairs(detect_positions, ground_truth, binded_index, overlap_score);
			return;
		}

		std::vector<int> idx;
		util::argsort_vector(scores, idx);
		std::vector<bool> gt_check(ground_truth.size(), false);
		binded_index.resize(detect_positions.size());
		overlap_score.resize(detect_positions.size());
		for (int i = idx.size() - 1; i >= 0; i--){
			int detect_id = idx[i];
			cv::Rect detect_rect = detect_positions[detect_id];
			float max_score = overlap_threshold;
			int max_j = -1;
			for (int j = 0; j < ground_truth.size(); j++){
				if (gt_check[j])
					continue;
				float ol_score = calcRectOverlap(detect_rect, ground_truth[j]);
				if (ol_score > max_score){
					max_score = ol_score;
					max_j = j;
				}
			}
			binded_index[detect_id] = max_j;
			overlap_score[detect_id] = 0;
			if (max_j >= 0){
				gt_check[max_j] = true;
				overlap_score[detect_id] = max_score;
			}
		}
	}
	

	// 検出オブジェクトとGround Truthデータを結びつける
	void bindRectPairs(const std::vector<std::vector<cv::Rect>>& detect_positions,
		const std::vector<std::vector<cv::Rect>>& ground_truth, 
		const std::vector<std::vector<float>>& scores,
		std::vector<std::vector<int>>& binded_index, 
		std::vector<std::vector<float>>& overlap_score,
		int* ground_truth_num, float overlap_threshold)
	{
		assert(detect_positions.size() == ground_truth.size());
		assert(scores.empty() || scores.size() == ground_truth.size());

		int N = detect_positions.size();
		binded_index.resize(N);
		overlap_score.resize(N);
		*ground_truth_num = 0;
		for (int n = 0; n < N; n++){
			if (scores.empty()){
				bindRectPairs(detect_positions[n], ground_truth[n], binded_index[n], overlap_score[n]);
			}
			else{
				bindRectPairs(detect_positions[n], ground_truth[n], scores[n], binded_index[n], overlap_score[n], overlap_threshold);
			}
			*ground_truth_num += ground_truth[n].size();
		}
	}

	
/*	// 検出オブジェクトとGround Truthデータを結びつける
	void bindRectPairs(const std::vector<std::vector<cv::Rect>>& detect_positions,
		const std::vector<std::vector<cv::Rect>>& ground_truth,
		std::vector<std::vector<int>>& binded_index,
		std::vector<std::vector<float>>& overlap_score,
		int* ground_truth_num)
	{
		assert(detect_positions.size() == ground_truth.size());

		int N = detect_positions.size();
		binded_index.resize(N);
		overlap_score.resize(N);
		*ground_truth_num = 0;
		for (int n = 0; n < N; n++){
			bindRectPairs(detect_positions[n], ground_truth[n], binded_index[n], overlap_score[n]);
			*ground_truth_num += ground_truth[n].size();
		}
	}
	*/

	float AveragePrecision(std::vector<float>& precision, std::vector<float>& recall)
	{
		assert(precision.size() == recall.size());
		
		std::vector<int> sort_idx;
		util::argsort_vector(recall, sort_idx);

		float thresholds[11];
		for (int t = 0; t <= 10; t++){
			thresholds[t] = 0.1 * t;
		}

		int t = 0;
		float total_p = 0.0;
		for (int i = 0; i < sort_idx.size(); i++){
			int idx = sort_idx[i];
			if (recall[idx] >= thresholds[t]){
				total_p += precision[idx];
				t++;
			}
		}
		if (t > 0)
			return total_p / t;
		else
			return 0;
	}



	float RecallPrecision(const std::vector<std::vector<int>>& binded_idx, 
		const std::vector<std::vector<float>>& scores, 
		const std::vector<std::vector<float>>& overlap_score,
		int ground_truth_num,
		std::vector<float>& recall, std::vector<float>& precision, std::vector<float>& thresholds,
		float overlap_threshold)
	{
		if (scores.empty())
			return -1;

		assert(binded_idx.size() == overlap_score.size());
		assert(scores.size() == binded_idx.size());

		std::vector<float> score_list;
		for (int i = 0; i < binded_idx.size(); i++){
			score_list.insert(score_list.end(), scores[i].begin(), scores[i].end());
		}

		std::sort(score_list.begin(), score_list.end());
		float prev = 0;
		for (int i = 0; i < score_list.size(); i++){
			float th = score_list[i];
			if (th == prev)
				continue;
			float th2 = (th + prev) / 2;
			std::vector<std::vector<int>> true_positive_id, false_positive_id;
			
			EvaluateDetection(binded_idx, scores, overlap_score, true_positive_id, false_positive_id, th2, overlap_threshold);

			int true_positive_num = util::CountVectorElements(true_positive_id);
			int false_positive_num = util::CountVectorElements(false_positive_id);

			prev = th;
			thresholds.push_back(th2);
			recall.push_back((float)true_positive_num / ground_truth_num);
			precision.push_back((float)true_positive_num / (true_positive_num + false_positive_num));
		}

		return AveragePrecision(precision, recall);
	}


	void EvaluateDetection(const std::vector<int>& binded_index, const std::vector<float>& overlap_scores,
		std::vector<int>& true_positive_id, std::vector<int>& false_positive_id, float overlap_threshold)
	{
		assert(binded_index.size() == overlap_scores.size());

		for (int i = 0; i < binded_index.size(); i++){
			if (binded_index[i] >= 0 && overlap_scores[i] > overlap_threshold)
				true_positive_id.push_back(i);
			else
				false_positive_id.push_back(i);
		}
	}


	void EvaluateDetection(const std::vector<int>& binded_index, const std::vector<float>& scores,
		const std::vector<float>& overlap_scores,
		std::vector<int>& true_positive_id, std::vector<int>& false_positive_id,
		float threshold, float overlap_threshold)
	{
		assert(binded_index.size() == overlap_scores.size());
		assert(scores.size() == binded_index.size());

		for (int i = 0; i< binded_index.size(); i++){
			if (scores[i] > threshold){
				if (binded_index[i] >= 0 && overlap_scores[i] > overlap_threshold)
					true_positive_id.push_back(i);
				else
					false_positive_id.push_back(i);
			}
		}
	}


	void EvaluateDetection(const std::vector<std::vector<int>>& binded_index,
		const std::vector<std::vector<float>>& scores,
		const std::vector<std::vector<float>>& overlap_scores,
		std::vector<std::vector<int>>& true_positive_id, 
		std::vector<std::vector<int>>& false_positive_id,
		float threshold,
		float overlap_threshold)
	{
		assert(binded_index.size() == overlap_scores.size());
		assert(scores.empty() || scores.size() == binded_index.size());

		int N = binded_index.size();
		true_positive_id.resize(N);
		false_positive_id.resize(N);
		int total_true_positive_num = 0;
		for (int n = 0; n < binded_index.size(); n++){
			if (scores.empty()){
				EvaluateDetection(binded_index[n], overlap_scores[n], true_positive_id[n], false_positive_id[n], overlap_threshold);
			}
			else{
				EvaluateDetection(binded_index[n], scores[n], overlap_scores[n], true_positive_id[n], false_positive_id[n], threshold, overlap_threshold);
			}
		}
	}


	void EvaluateAll(const std::vector<std::vector<cv::Rect>>& detect_positions,
		const std::vector<std::vector<cv::Rect>>& ground_truth,
		std::vector<std::vector<cv::Rect>> &true_positives, std::vector<std::vector<cv::Rect>> &false_positives,
		float overlap_th)
	{
		std::vector<std::vector<int>> binded_index;
		std::vector<std::vector<float>> overlap_score;
		int ground_truth_num;
		bindRectPairs(detect_positions, ground_truth, binded_index, overlap_score, &ground_truth_num);

		std::vector<std::vector<int>> true_positive_id, false_positive_id;
		EvaluateDetection(binded_index, overlap_score, true_positive_id, false_positive_id, overlap_th);
		Id2Positions(detect_positions, true_positive_id, true_positives);
		Id2Positions(detect_positions, false_positive_id, false_positives);
	}


	void EvaluateAll(const std::vector<std::vector<cv::Rect>>& detect_positions, 
		const std::vector<std::vector<cv::Rect>>& ground_truth, 
		const std::vector<std::vector<float>>& scores, 
		float threshold,
		std::vector<float>& recall, std::vector<float>& precision, std::vector<float>& thresholds, 
		std::vector<std::vector<cv::Rect>>& true_positives, std::vector<std::vector<cv::Rect>>& false_positives,
		float* average_precision,
		float overlap_th)
	{
		std::vector<std::vector<int>> binded_index;
		std::vector<std::vector<float>> overlap_score;
		int ground_truth_num;
		//bindRectPairs(detect_positions, ground_truth, binded_index, overlap_score, &ground_truth_num);
		bindRectPairs(detect_positions, ground_truth, scores, binded_index, overlap_score, &ground_truth_num, overlap_th);

		std::vector<std::vector<int>> true_positive_id, false_positive_id;
		EvaluateDetection(binded_index, scores, overlap_score, true_positive_id, false_positive_id, threshold, overlap_th);
		Id2Positions(detect_positions, true_positive_id, true_positives);
		Id2Positions(detect_positions, false_positive_id, false_positives);
		*average_precision = RecallPrecision(binded_index, scores, overlap_score, ground_truth_num, recall, precision, thresholds);
	}


	void ThresholdDetectPositions(const std::vector<cv::Rect>& detect_positions,
		const std::vector<float>& scores, float threshold,
		std::vector<cv::Rect>& output_positions)
	{
		assert(detect_positions.size() == scores.size());
		for (int i = 0; i < detect_positions.size(); i++){
			if (scores[i] > threshold)
				output_positions.push_back(detect_positions[i]);
		}
	}

	
	void ThresholdDetectPositions(const std::vector<std::vector<cv::Rect>>& detect_positions,
		const std::vector<std::vector<float>>& scores, float threshold,
		std::vector<std::vector<cv::Rect>>& output_positions)
	{
		assert(detect_positions.size() == scores.size());
		int test_num = detect_positions.size();
		output_positions.clear();
		output_positions.resize(test_num);
		for (int i = 0; i < test_num; i++){
			ThresholdDetectPositions(detect_positions[i], scores[i], threshold, output_positions[i]);
		}
	}


	void Id2Positions(const std::vector<cv::Rect>& all_positions,
		const std::vector<int>& position_id, std::vector<cv::Rect>& positions)
	{
		positions.clear();

		std::vector<int>::const_iterator it = position_id.begin(),
			it_e = position_id.end();
		while (it != it_e){
			positions.push_back(all_positions[*it]);
			it++;
		}
	}
	

	void Id2Positions(const std::vector<std::vector<cv::Rect>>& all_positions,
		const std::vector<std::vector<int>>& position_id, std::vector<std::vector<cv::Rect>>& positions)
	{
		int N = all_positions.size();
		positions.resize(N);
		for (int i = 0; i < N; i++){
			Id2Positions(all_positions[i], position_id[i], positions[i]);
		}
	}

}
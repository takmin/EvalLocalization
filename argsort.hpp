
#include <algorithm>

namespace util{

	template <typename T>
	struct ARG_SORTER
	{
		T	val;
		int idx;

		bool operator<(const struct ARG_SORTER& right) const{
			return val < right.val;
		}
	};

	template <typename T>
	void argsort_vector(const std::vector<T>& vec, std::vector<int>& idx)
	{
		int vec_size = vec.size();
		std::vector<struct ARG_SORTER<T>> sort_pairs;
		for (int i = 0; i < vec_size; i++){
			struct ARG_SORTER<T> argsorter;
			argsorter.val = vec[i];
			argsorter.idx = i;
			sort_pairs.push_back(argsorter);
		}

		std::sort(sort_pairs.begin(), sort_pairs.end());

		idx.clear();
		for (int i = 0; i < vec_size; i++){
			idx.push_back(sort_pairs[i].idx);
		}
	}

}
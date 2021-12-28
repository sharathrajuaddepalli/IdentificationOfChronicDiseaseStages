#include<iostream>
#include<vector>
#include<cmath>
#include <stdlib.h>     
#include <time.h>       
#include <algorithm>      
#include<string>
#include<fstream>
#include <sstream>

using namespace std;

class Utils {
public:
	Utils() {

	}
               
               static double getDistance(vector<double> x1, vector<double> x2) {
		double sum = 0;
		for (int i = 0; i < x1.size(); i++)
			sum += pow(x1[i] - x2[i], 2);
		return sqrt(sum);
	}

	static vector<vector<double>> matT(vector<vector<double>> a) {
		vector<vector<double>> aT;

		for (int i = 0; i < a[0].size(); i++)
		{
			vector<double> row;
			for (int j = 0; j < a.size(); j++)
			{
				row.push_back(a[j][i]);
			}
			aT.push_back(row);
		}

		return aT;
	}

	static vector<vector<double>> matDot(vector<vector<double>> a, vector<vector<double>> b) {

		vector<vector<double>> dMat;

		vector<vector<double>> bT = Utils::matT(b);

		for (int i = 0; i < a.size(); i++)
		{
			vector<double> n_row;

			vector<double> row = a[i];

			for (int j = 0; j < bT.size(); j++)
			{
				vector<double> col = bT[j];

				double sum = 0;
				for (int k = 0; k < row.size(); k++)
					sum += (row[k] * col[k]);

				n_row.push_back(sum);
			}
			dMat.push_back(n_row);
		}

		return dMat;
	}

	static double argmax(vector<double> a) {
		double maxE = -999, maxIdx = 0;
		for (int i = 0; i < a.size(); i++) {
			if (a[i] > maxE) {
				maxE = a[i];
				maxIdx = i;
			}
		}
		return maxIdx;
	}
	
	static vector<vector<double>> col(vector<vector<double>> A, int i) {
		vector<double> c;
		for (int j = 0; j < A.size(); j++)
			c.push_back(A[j][i]);
		vector<vector<double>> n;
		n.push_back(c);
		return n;
	}

	static vector<vector<double>> add2Mat(
		vector<vector<double>> A,
		vector<vector<double>> B,
		double coeff) {


		vector<vector<double>> C;

		for (size_t i = 0; i < A.size(); i++)
		{
			vector<double> row;
			for (size_t j = 0; j < A[0].size(); j++)
			{
				row.push_back( (A[i][j] + coeff*B[i][j]));
			}
			C.push_back(row);
		}
		return C;
	}

	static double getMatrixSum(vector<vector<double>> c) {
		double sum = 0;
		for (int i = 0; i < c.size(); i++)
			for (int j = 0; j < c[0].size(); j++)
				sum += c[i][j];
		return sum;
	}

	static vector<vector<double>> convert_to_one_hot(vector<double> vec,int num_of_classes) {
		vector<vector<double>> A;

		for (int i = 0; i < vec.size(); i++)
		{
			vector<double> row(num_of_classes, 0.0);
			row[(int)vec[i]] = 1.0;
			A.push_back(row);
		}
		return A;
	}

	static vector<vector<double>> inv(vector<vector<double>> a) {


		int n = a.size(), i = 0, j = 0, k = 0;

		vector<vector<double>> b;
		vector<double> p(n, 0.0);

		for (i = 0; i < n; i++) {
			vector<double> row(n, 0.0);
			b.push_back(row);
		}


		for (i = 0; i < n; i++)
		{
			for (j = 0; j < n; j++)
			{
				if (i == j)
				{
					b[i][j] = 1;
				}
				else
				{
					b[i][j] = 0;
				}
			}
		}

		for (i = 0; i < n; i++)
		{
			p[i] = a[i][i];
			for (j = 0; j < n; j++)
			{
				b[i][j] = b[i][j] / p[i];
				a[i][j] = a[i][j] / p[i];
			}
			for (j = 0; j < n; j++)
			{
				for (k = 0; k < n; k++)
				{
					if (j != i)
					{
						p[j] = a[j][i];
						b[j][k] -= b[i][k] * p[j];
					}
				}
			}
			for (j = 0; j < n; j++)
			{
				for (k = 0; k < n; k++)
				{
					if (j != i)
					{
						a[j][k] -= a[i][k] * p[j];
					}
				}
			}

		}

		return b;

	}

};

class kmeans {

public:
	vector<vector<double>> centroids;
	vector<vector<vector<double>>> cluster_list;

	kmeans(vector<vector<double>> X, int k, int max_iters) {
		this->X = X;
		this->k = k;
		this->max_iters = max_iters;
	}


	kmeans fit() {


		for (int i = 0; i < k; i++) {
			vector<double> cetroid;
			for (int j = 0; j < X[0].size(); j++)
				cetroid.push_back((double)(rand() % 100));
			centroids.push_back(cetroid);
		}

		bool converged = false;
		int current_iter = 0;

		while (!converged && current_iter < max_iters) {

			cluster_list.clear();
			for (int i = 0; i < centroids.size(); i++) {
				vector<vector<double>> a;
				cluster_list.push_back(a);
			}

			for (vector<double> x : X) {
				vector<double> distances_list;
				for (vector<double> c : centroids)
					distances_list.push_back(Utils::getDistance(x, c));
				int minIdex = min_element(distances_list.begin(), distances_list.end()) - distances_list.begin();
				cluster_list[minIdex].push_back(x);
			}

			// Remove empty clusters
			vector<int> emptyIndexes;
			for (int i = 0; i < cluster_list.size(); i++)
				if (cluster_list[i].empty())
					emptyIndexes.push_back(i);

			for (int i = 0; i < emptyIndexes.size(); i++)
				cluster_list.erase(cluster_list.begin() + (emptyIndexes[i] - i));

			vector<vector<double>> prev_centroids;
			for (vector<double> c : centroids)
				prev_centroids.push_back(c);

			centroids.clear();

			for (int i = 0; i < cluster_list.size(); i++)
				centroids.push_back(get_column_mean(cluster_list[i]));

			double pattern = abs(Utils::getMatrixSum(centroids) - Utils::getMatrixSum(prev_centroids));

			cout << "K-MEANS: " << (int)pattern << endl;

			converged = pattern == 0;
			current_iter++;

		}

		return *this;
	}


private:
	vector<vector<double>> X;
	int k;
	int max_iters;

	vector<double> get_column_mean(vector<vector<double>> cluster) {

		vector<double> n_centroid;

		for (int m = 0; m < cluster[0].size(); m++) {
			double sum = 0;
			for (int n = 0; n < cluster.size(); n++)
				sum += cluster[n][m];
			n_centroid.push_back(sum / cluster.size());
		}

		return n_centroid;
	}

	


};

class RBF {

public:
	vector<vector<double>> w;
	double acc = 0;

	RBF(vector<vector<double>> trX,
		vector<double> trY,
		vector<vector<double>> tsX,
		vector<double> tsY,
		int num_of_classes, int k) {

		this->trX = trX;
		this->trY = trY;
		this->tsX = tsX;
		this->tsY = tsY;
		this->num_of_classes = num_of_classes;
		this->k = k;

	}

	


	double get_rbf(vector<double> x, vector<double> c, double s) {
		double distance = Utils::getDistance(x, c);
		return 1 / exp(-distance / (s * s));
	}


	RBF fit() {

		// get centroids
		vector<vector<double>> centroids = kmeans(trX,k, 100).fit().centroids;

	
		// get stds
		double dmax = 0;
		for (vector<double> c : centroids) {
			for (vector<double>x : trX) {
				double d = Utils::getDistance(c, x);
				if (d > dmax)
					dmax = d;
			}
		}
		double std = dmax / sqrt(2 * k);

		
		vector<vector<double>> RBF_X = getAsRbfList(trX, centroids, std);
		
		vector<vector<double>> hot_tr_y = Utils::convert_to_one_hot(trY,num_of_classes);

		vector<vector<double>> RBF_X_T = Utils::matT(RBF_X);

		vector< vector<double>> w
			= Utils::matDot(Utils::matDot(Utils::inv(Utils::matDot(RBF_X_T, RBF_X)), RBF_X_T), hot_tr_y);

		acc = getAcc(tsX, tsY, w, centroids, std);

		 return *this;
	}


private:

	vector<vector<double>> trX, tsX;
	vector<double> trY, tsY;
	int num_of_classes, k;

	double getAcc(vector<vector<double>> X, vector<double> y,
		vector<vector<double>>w, vector<vector<double>>centroids, double std
		) {

		vector<vector<double>> ts_rbf_list = getAsRbfList(X, centroids, std);
	

		vector<vector<double>> pred_test_y_one_hot = Utils::matDot(ts_rbf_list, w);

		vector<double> pred_test_y;
		for (vector<double> row : pred_test_y_one_hot)
			pred_test_y.push_back(Utils::argmax(row));

		
		double true_counter = 0;

		for (int i = 0; i < pred_test_y.size(); i++)
			if (pred_test_y[i] ==y[i])
				true_counter++;

		return true_counter / pred_test_y.size();

	}


	vector<vector<double>> getAsRbfList(vector<vector<double>> X
	, vector<vector<double>> centroids, double std) {

		// get RBFs
		vector<vector<double>> rbf_list;
		for (vector<double> x : X) {
			vector<double> rbf_row;
			for (vector<double> c : centroids) {
				rbf_row.push_back(get_rbf(x, c, std));
			}
			rbf_list.push_back(rbf_row);
		}
		return rbf_list;
	}



};

void print2Dvector(vector<vector<double>> vec) {
	for (vector<double> row : vec) {
		for (double a : row)
			cout << a << " ";
		cout << endl;
	}
}

vector<vector<double>> read_record(string path)
{
	vector<vector<double>> data;
	ifstream file(path);
	string str;
	while (getline(file, str))
	{
		stringstream ss(str);
		
		vector<double> row;

		while (ss.good())
		{
			string substr;
			getline(ss, substr, ',');
			row.push_back(stod(substr));
		}
		data.push_back(row);
	}

	return data;
}


int main() {
	
	srand(time(NULL));

	vector<vector<double>> data = read_record("C:\Users\Pranav\Desktop\RBF\NewRBFdataset.csv");
		
	
	vector<vector<double>> train_x;
	vector<vector<double>> test_x;

	vector<double>train_y;
	vector<double>test_y;

	int train_count = 300;
	int test_count = 100;
	int k = 500;

	for (int i = 0; i < train_count; i++)
	{
		vector<double> row;
		for (int j = 0; j < data[0].size(); j++)
		{
			if (j == 0)
				train_y.push_back(data[i][j]);
			else
				row.push_back(data[i][j]);
		}
		train_x.push_back(row);
	}

	


	for (int i = train_count; i < train_count+test_count; i++)
	{
		vector<double> row;
		for (int j = 0; j < data[0].size(); j++)
		{
			if (j == 0)
				test_y.push_back(data[i][j]);
			else
				row.push_back(data[i][j]);
		}
		test_x.push_back(row);
	}


	double num_of_classes = *max_element(begin(train_y), end(train_y)) + 1;
	 
	RBF classifier = RBF(train_x, train_y, test_x, test_y, num_of_classes, k).fit();
	cout << endl << "Accuracy: " << classifier.acc << endl;
	
	

	system("PAUSE");

         return 0;

}
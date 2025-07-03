#include<iostream>
#include<cmath>
#include<fstream>
#include<vector>
using namespace std;
template<class T>
class matrix
{
public:
	int row,col,size;
	T* arr;
	matrix():row(0),col(0),size(0),arr(nullptr) {}
	matrix<T>(int r,int c,T num = T()) : row(r),col(c),size(r* c),arr(new T[r * c])
	{
		for(int i = 0;i < size;++i)arr[i] = num;
	}
	matrix<T>(const matrix<T>& m) : row(m.row),col(m.col),size(m.size),arr(new T[m.size])
	{
		for(int i = 0;i < size;++i)arr[i] = m.arr[i];
	}
	matrix<T>(matrix<T>&& m)noexcept:row(m.row),col(m.col),size(m.size),arr(m.arr)
	{
		m.arr = nullptr;
		m.row = m.col = m.size = 0;
	}
	matrix<T>& operator=(const matrix<T>& m)//copy
	{
		if(this == &m)return *this;
		if(row != m.row || col != m.col)
		{
			delete[] arr;
			row = m.row;
			col = m.col;
			size = m.size;
			arr = new T[size];
		}
		for(int i = 0;i < size;++i)arr[i] = m.arr[i];
		return *this;
	}
	matrix<T>& operator=(matrix<T>&& m) noexcept
	{
		if(this != &m)
		{
			delete[] arr; // 不论尺寸如何，先释放当前资源
			row = m.row;
			col = m.col;
			size = m.size;
			arr = m.arr;
			m.arr = nullptr;
			m.row = m.col = m.size = 0;
		}
		return *this;
	}
	matrix<T> operator+(const T& num)
	{
		matrix<T> res(*this);
		for(int i = 0;i < res.size;++i)res.arr[i] += num;
		return res;
	}
	friend matrix<T> operator+(const T& num,const matrix<T>& m)
	{
		matrix<T>res(m);
		for(int i = 0;i < m.size;++i)res.arr[i] += num;
		return res;
	}
	matrix<T> operator+(const matrix<T>& m)
	{
		if(row != m.row || col != m.col)
		{
			cout << "Matrix size not matching" << endl;
			return *this;
		}
		matrix<T> res(*this);
		for(int i = 0;i < res.size;++i)res.arr[i] += m.arr[i];
		return res;
	}
	matrix<T>& operator+=(const matrix<T>& m)
	{
		if(row != m.row || col != m.col)
		{
			cout << "Matrix size not matching" << endl;
			return *this;
		}
		for(int i = 0;i < size;++i)arr[i] += m.arr[i];
		return *this;
	}
	matrix<T>& operator+=(const T& num)
	{
		for(int i = 0;i < size;++i)arr[i] += num;
		return *this;
	}
	matrix<T> operator-()
	{
		matrix<T> res(*this);
		for(int i = 0;i < res.size;++i)res.arr[i] = -arr[i];
		return res;
	}
	matrix<T> operator-(const T& num)
	{
		matrix<T> res(*this);
		for(int i = 0;i < size;++i)res.arr[i] -= num;
		return res;
	}
	matrix<T> operator-(const matrix<T>& m)
	{
		if(row != m.row || col != m.col)
		{
			cout << "Matrix size not matching" << endl;
			return *this;
		}
		matrix<T> res(*this);
		for(int i = 0;i < res.size;++i)res.arr[i] -= m.arr[i];
		return res;
	}
	matrix<T>& operator-=(const matrix<T>& m)
	{
		if(row != m.row || col != m.col)
		{
			cout << "Matrix size not matching" << endl;
			return *this;
		}
		for(int i = 0;i < size;++i)arr[i] -= m.arr[i];
		return *this;
	}
	matrix<T>& operator-=(const T& num)
	{
		for(int i = 0;i < size;++i)arr[i] -= num;
		return *this;
	}
	matrix<T> operator*(const T& num)
	{
		matrix<T> res(*this);
		for(int i = 0;i < size;++i)res.arr[i] *= num;
		return res;
	}
	friend matrix<T> operator*(const T& num,const matrix<T>& m)
	{
		matrix<T> res(m);
		for(int i = 0;i < m.size;++i)res.arr[i] *= num;
		return res;
	}
	matrix<T> operator*(const matrix<T>& m)
	{
		if(col != m.row)
		{
			cout << "Matrix size not matching" << endl;
			return *this;
		}
		matrix<T> res(row,m.col);
		for(int i = 0;i < row;++i)
			for(int j = 0;j < m.col;++j)
				for(int k = 0;k < col;++k)
					res.arr[i * m.col + j] += arr[i * col + k] * m.arr[k * m.col + j];
		return res;
	}
	matrix<T>& operator*=(const T& num)
	{
		for(int i = 0;i < size;++i)arr[i] *= num;
		return *this;
	}
	matrix<T>& operator*=(const matrix<T>& m)
	{
		*this = (*this) * m;
		return *this;
	}
	matrix<T> operator^(const matrix<T>& m)//逐元素相乘
	{
		if(row != m.row || col != m.col)
		{
			cout << "Matrix size not matching" << endl;
			return *this;
		}
		matrix<T> res(*this);
		for(int i = 0;i < size;++i)res.arr[i] *= m.arr[i];
		return res;
	}
	matrix<T>& operator^=(const matrix<T>& m)
	{
		if(row != m.row || col != m.col)
		{
			cout << "Matrix size not matching" << endl;
			return *this;
		}
		for(int i = 0;i < size;++i)arr[i] *= m.arr[i];
		return *this;
	}
	matrix<T> transpose()
	{
		matrix<T> res(col,row);
		for(int i = 0;i < row;++i)
			for(int j = 0;j < col;++j)
				res.arr[j * row + i] = arr[i * col + j];
		return res;
	}
	template<class F>
	matrix<T> func(F f)//逐元素一元函数
	{
		matrix<T> res(row,col);
		for(int i = 0;i < size;++i)res.arr[i] = f(arr[i]);
		return res;
	}
	template<class F>
	matrix<T> func(F f,const matrix<T>& m)//逐元素二元函数
	{
		if(row != m.row || col != m.col)
		{
			cout << "Matrix size not matching" << endl;
			return *this;
		}
		matrix<T>res(row,col);
		for(int i = 0;i < size;++i)res.arr[i] = f(arr[i],m.arr[i]);
		return res;
	}
	matrix<T> flatten()
	{
		matrix<T> res(*this);
		res.row = size;
		res.col = 1;
		return res;
	}
	matrix<T> extend_col(int n)//[row,1]->[row,n]
	{
		if(col != 1)
		{
			cout << "Matrix size not matching" << endl;
			return *this;
		}
		matrix<T>res(row,n);
		for(int i = 0;i < row;++i)
			for(int j = 0;j < n;++j)res.arr[i * n + j] = arr[i];
		return res;
	}
	matrix<T> extend_row(int n)//[1,col]->[n,col]
	{
		if(row != 1)
		{
			cout << "Matrix size not matching" << endl;
			return *this;
		}
		matrix<T>res(n,col);
		for(int i = 0;i < n;++i)
			for(int j = 0;j < col;++j)res.arr[i * col + j] = arr[j];
		return res;
	}
	matrix<T> max_col()//每一列的最大值组成的矩阵 [row,col]->[1,col]
	{
		matrix<T> res(1,col);
		for(int i = 0;i < col;++i)
		{
			T maxn = arr[i];
			for(int j = 1;j < row;++j)maxn = max(maxn,arr[j * col + i]);
			res.arr[i] = maxn;
		}
		return res;
	}
	T sum()
	{
		T res = T();
		for(int i = 0;i < size;++i)res += arr[i];
		return res;
	}
	matrix<T> sum_row()//行求和 [row,col]->[row,1]
	{
		matrix<T> res(row,1);
		for(int i = 0;i < row;++i)
		{
			T sum = T();
			for(int j = 0;j < col;++j)sum += arr[i * col + j];
			res.arr[i] = sum;
		}
		return res;
	}
	matrix<T> sum_col()//列求和 [row,col]->[1,col]
	{
		matrix<T> res(1,col);
		for(int i = 0;i < col;++i)
		{
			T sum = T();
			for(int j = 0;j < row;++j)sum += arr[j * col + i];
			res.arr[i] = sum;
		}
		return res;
	}
	T* operator[](int i)
	{
		return arr + (i * col);
	}
	friend ostream& operator<<(ostream& os,const matrix<T>& m)
	{
		for(int i = 0;i < m.row;++i)
		{
			for(int j = 0;j < m.col;++j)
			{
				os << m.arr[i * m.col + j] << " ";
			}
			os << endl;
		}
		return os;
	}
	~matrix()
	{
		if(arr)
		{
			delete[]arr;
		}
	}
};
class node
{
public:
	virtual matrix<double> forward(matrix<double>& input) = 0;
	virtual matrix<double> backward(matrix<double>& grad_output) = 0;
	matrix<double> last_input;
	virtual void save_parameters(ofstream& ofs) = 0;
	virtual void load_parameters(ifstream& ifs) = 0;
	virtual double l1_loss() = 0;
	virtual double l2_loss() = 0;
};
class Parameter
{
public:
	static double lr,wd1,wd2;
	static int batch_size,max_epoch;
};
//1e-2 17751 1e-3 15550 1e-1 1943 8e-3 17667 2e-3 4000 89/100
//double& wd1 = MNIST::wd1;
//double& wd2 = MNIST::wd2;
class Relu:public node
{
public:
	matrix<double> forward(matrix<double>& input)
	{
		last_input = input;
		return input.func([](double x) { return x > 0 ? x : 0; });
	}
	matrix<double> backward(matrix<double>& grad_output)
	{
		return grad_output.func([&](double x,double y) { return y > 0 ? x : 0; },last_input);
	}
	void save_parameters(ofstream& ofs) {}
	void load_parameters(ifstream& ifs) {}
	double l1_loss() { return 0; }
	double l2_loss() { return 0; }
};
class Linear:public node
{
public:
	int in_row,out_row;
	matrix<double> weight,bias;//weight:[in_row,out_row] bias:[out_row,1]
	Linear(int in_row,int out_row):in_row(in_row),out_row(out_row),weight(in_row,out_row),bias(out_row,1)
	{
		for(int i = 0;i < in_row;++i)
			for(int j = 0;j < out_row;++j)
				weight[i][j] = ((rand() % 1000) / 1000.0 - 0.5) * 0.1;
	}
	matrix<double>forward(matrix<double>& input)//input:[in_row,batch]
	{
		last_input = input;
		return weight.transpose() * input + bias.extend_col(input.col);
	}
	matrix<double> backward(matrix<double>& grad_output)//grad_output:[out_row,batch]
	{
		bias -= grad_output.sum_row() * Parameter::lr;
		weight -= last_input * grad_output.transpose() * Parameter::lr
			+ weight * Parameter::wd2 * Parameter::lr
			+ weight.func([](double x)
		{
			return x > 0 ? 1.0 : (x < 0 ? -1.0 : 0.0);
		}) * Parameter::wd1 * Parameter::lr;
		//double max_grad = 0.5; // 阈值
		//for(int i = 0; i < weight.size; ++i)
		//{
		//	if(weight.arr[i] > max_grad) weight.arr[i] = max_grad;
		//	if(weight.arr[i] < -max_grad) weight.arr[i] = -max_grad;
		//}
		return weight * grad_output;//grad_input:[in_row,batch]
	}
	void save_parameters(ofstream& ofs)
	{
		for(int i = 0;i < weight.size;++i)ofs.write((char*)(weight.arr + i),sizeof(double));
		for(int i = 0;i < bias.size;++i)ofs.write((char*)(bias.arr + i),sizeof(double));
	}
	void load_parameters(ifstream& ifs)
	{
		for(int i = 0;i < weight.size;++i)ifs.read((char*)(weight.arr + i),sizeof(double));
		for(int i = 0;i < bias.size;++i)ifs.read((char*)(bias.arr + i),sizeof(double));
	}
	double l1_loss()
	{
		return weight.func([](double x) { return abs(x); }).sum();
	}
	double l2_loss()
	{
		return (weight ^ weight).sum();
	}
};
class Sigmoid:public node
{
	static double sigmoid(double x)
	{
		return 1 / (1 + exp(-x));
	}
	static double dsigmoid(double x)
	{
		return sigmoid(x) * sigmoid(-x);
	}
public:
	matrix<double> forward(matrix<double>& input)
	{
		last_input = input;
		return input.func(sigmoid);
	}
	matrix<double> backward(matrix<double>& grad_output)
	{
		return grad_output ^ last_input.func(dsigmoid);
	}
	void save_parameters(ofstream& ofs) {}
	void load_parameters(ifstream& ifs) {}
	double l1_loss() { return 0; }
	double l2_loss() { return 0; }
};
class Softmax:public node
{
public:
	matrix<double> forward(matrix<double>& input)//input:[in_row,batch]
	{
		matrix<double> shifted = (input - input.max_col().extend_row(input.row))
			.func([](double x) { return exp(x); });
		//shifted = shifted.func([](double x) { return (x < -20) ? -20 : (x > 20) ? 20 : x; });
		matrix<double> sum = shifted.sum_col().extend_row(input.row);
		for(int i = 0; i < sum.size; ++i)
		{
			if(sum.arr[i] == 0 || std::isnan(sum.arr[i]) || std::isinf(sum.arr[i]))
				sum.arr[i] = 1e-9;
		}
		last_input = shifted.func(
			[](double x,double y) { return x / y; },
			sum    // ← 一定要用这个经过修正的 sum！
		);
		return last_input;
	}
	matrix<double> backward(matrix<double>& grad_output)//grad_output:[out_row,batch]
	{
		//return grad_output ^ (-last_input + 1) ^ last_input;
		//return grad_output;
		return grad_output;
	}
	void save_parameters(ofstream& ofs) {}
	void load_parameters(ifstream& ifs) {}
	double l1_loss() { return 0; }
	double l2_loss() { return 0; }
};
class CrossEntropy:public node
{
public:
	vector<int> label;
	double Loss = 0;
	CrossEntropy() {}
	matrix<double> forward(matrix<double>& input)
	{
		last_input = input;
		Loss = 0;
		for(int i = 0; i < label.size(); ++i)
		{
			double prob = input[label[i]][i];
			if(prob <= 0 || std::isnan(prob) || std::isinf(prob))
			{
				//printf("Invalid prob: %lf at label %d index %d\n",prob,label[i],i);
				prob = 1e-9;
			}
			Loss -= log(prob);
		}
		Loss /= label.size();
		return input;
	}
	matrix<double> backward(matrix<double>& grad_output)
	{
		matrix<double> res(last_input);
		int N = label.size();
		for(int i = 0; i < N; ++i)
		{
			res[label[i]][i] -= 1.0;
		}
		res *= 1.0 / N;
		return res;
	}
	void save_parameters(ofstream& ofs) {}
	void load_parameters(ifstream& ifs) {}
	double l1_loss() { return 0; }
	double l2_loss() { return 0; }
};
class Graph
{
public:
	node** nodes;
	int size = 11;
	Graph()
	{
		nodes = new node * [10];
		nodes[0] = new Linear(784,512);
		nodes[1] = new Relu();
		nodes[2] = new Linear(512,512);
		nodes[3] = new Relu();
		nodes[4] = new Linear(512,256);
		nodes[5] = new Relu();
		nodes[6] = new Linear(256,128);
		nodes[7] = new Relu();
		nodes[8] = new Linear(128,10);
		nodes[9] = new Softmax();
		nodes[10] = new CrossEntropy();
	}
	void save_parameters(string path)
	{
		ofstream ofs(path,ios::binary | ios::out);
		for(int i = 0;i < size;++i)nodes[i]->save_parameters(ofs);
	}
	void load_parameters(string path)
	{
		ifstream ifs(path,ios::binary | ios::in);
		for(int i = 0;i < size;++i)nodes[i]->load_parameters(ifs);
	}
	void showloss()
	{
		double L1_loss = 0,L2_loss = 0;
		for(int i = 0;i < size - 1;++i)L1_loss += nodes[i]->l1_loss(),L2_loss += nodes[i]->l2_loss();
		L1_loss *= Parameter::wd1;
		L2_loss *= Parameter::wd2;
		reinterpret_cast<CrossEntropy*>(nodes[size - 1])->Loss += L1_loss + L2_loss;
		printf("%.8lf Loss\n",reinterpret_cast<CrossEntropy*>(nodes[size - 1])->Loss);
	}
	matrix<double> run(const matrix<double>& input,vector<int> label)
	{
		reinterpret_cast<CrossEntropy*>(nodes[size - 1])->label = label;
		matrix<double> res = input;
		for(int i = 0;i < size;++i)
		{
			res = nodes[i]->forward(res);
		}
		//showloss();
		for(int i = size - 1;i >= 0;--i)
		{
			res = nodes[i]->backward(res);
			//cout << res << endl;
		}
		//delete nodes[size - 1];
		return res;
	}
	bool test(matrix<double>& input,vector<int> label)//input:[784,1]  res:[10,1] label:[1]
	{
		matrix<double> res = input;
		for(int i = 0;i < size - 1;++i)
		{
			res = nodes[i]->forward(res);
		}
		int max_label = 0;
		for(int i = 0;i < 10;++i)
		{
			if(res[i][0] > res[max_label][0])max_label = i;
		}
		if(max_label == label[0])return true;
		return false;
	}
	int predict(matrix<double>& input)//input:[784,1]  res:[10,1] label:[1]
	{
		matrix<double> res = input;
		for(int i = 0;i < size - 1;++i)
		{
			res = nodes[i]->forward(res);
		}
		int max_label = 0;
		for(int i = 0;i < 10;++i)
		{
			if(res[i][0] > res[max_label][0])max_label = i;
		}
		return max_label;
	}
	~Graph()
	{
		for(int i = 0;i < size;++i)
		{
			delete nodes[i];
		}
		delete[] nodes;
	}
};
int intread(ifstream& f)//大尾端和小尾端的问题！！！
{
	unsigned char tmp;
	int sum = 0;
	f.read((char*)&tmp,sizeof(tmp));
	sum += tmp << 24;
	f.read((char*)&tmp,sizeof(tmp));
	sum += tmp << 16;
	f.read((char*)&tmp,sizeof(tmp));
	sum += tmp << 8;
	f.read((char*)&tmp,sizeof(tmp));
	sum += tmp << 0;
	return sum;
}
class MNIST
{
public:
	matrix<double>* train_data,* test_data;
	vector<int>* train_label,* test_label;
	matrix<double>display_image = matrix<double>(784,1,0);
	Graph graph;
	void loadtraindata(string image_path = "C:\\Users\\pyh\\Desktop\\MNIST_data\\MNIST_data\\train-images.idx3-ubyte",string labels_path = "C:\\Users\\pyh\\Desktop\\MNIST_data\\MNIST_data\\train-labels.idx1-ubyte")
	{
		ifstream train_images(image_path,ios::binary | ios::in);
		ifstream train_labels(labels_path,ios::binary | ios::in);
		if(!train_images.is_open() || !train_labels.is_open())
		{
			cout << "MNIST data file not found" << endl;
			return;
		}int magic_image,magic_label,image_row,image_col,image_num,label_num;
		unsigned char tmp;
		int label;
		magic_image = intread(train_images) & 255;
		image_num = intread(train_images);
		image_row = intread(train_images);
		image_col = intread(train_images);
		magic_label = intread(train_labels);
		label_num = intread(train_labels);
		train_data = new matrix<double>[image_num / Parameter::batch_size + 5];
		train_label = new vector<int>[label_num / Parameter::batch_size + 5];
		for(int i = 0;i * Parameter::batch_size < image_num;++i)
		{
			train_data[i].row = image_row * image_col;
			train_data[i].col = Parameter::batch_size;
			train_data[i].size = train_data[i].row * train_data[i].col;
			train_data[i].arr = new double[train_data[i].row * train_data[i].col];
			for(int j = 0;j < Parameter::batch_size;++j)
			{
				for(int k = 0;k < image_row * image_col;++k)
				{
					train_images.read((char*)&tmp,sizeof(tmp));
					train_data[i][k][j] = tmp / 255.0;
				}
				train_labels.read((char*)&tmp,sizeof(tmp));
				train_label[i].push_back(tmp);
			}
			if(i % 50 == 0)cout << "batch:" << i << endl;
		}
		train_images.close();
		train_labels.close();
	}
	void train()
	{
		int num_batches = 60000 / Parameter::batch_size;
		for(int epoch = 0;epoch < Parameter::max_epoch;++epoch)
		{
			for(int i = 0;i < num_batches;++i)
			{
				graph.run(train_data[i],train_label[i]);
				if(i % 50 == 0)cout << "batch:" << i << endl;
			}
			//cout << "epoch:" << epoch << endl;
		}
	}
	void loadtestdata(string image_path = "C:\\Users\\pyh\\Desktop\\MNIST_data\\MNIST_data\\t10k-images.idx3-ubyte",string labels_path = "C:\\Users\\pyh\\Desktop\\MNIST_data\\MNIST_data\\t10k-labels.idx1-ubyte")
	{
		ifstream test_images(image_path,ios::binary | ios::in);
		ifstream test_labels(labels_path,ios::binary | ios::in);
		if(!test_images.is_open() || !test_labels.is_open())
		{
			cout << "MNIST data file not found" << endl;
			return;
		}int magic_image,magic_label,image_row,image_col,image_num,label_num;
		unsigned char tmp;
		magic_image = intread(test_images) & 255;
		image_num = intread(test_images);
		image_row = intread(test_images);
		image_col = intread(test_images);
		magic_label = intread(test_labels);
		label_num = intread(test_labels);
		cout << "image_num:" << image_num << endl;
		test_data = new matrix<double>[image_num + 5];
		test_label = new vector<int>[label_num + 5];
		for(int i = 0;i < image_num;++i)
		{
			test_data[i].row = image_row * image_col;
			test_data[i].col = 1;
			test_data[i].size = test_data[i].row * test_data[i].col;
			test_data[i].arr = new double[test_data[i].row * test_data[i].col];
			for(int k = 0;k < image_row * image_col;++k)
			{
				test_images.read((char*)&tmp,sizeof(tmp));
				test_data[i][k][0] = tmp / 255.0;
			}
			test_labels.read((char*)&tmp,sizeof(tmp));
			test_label[i].push_back(tmp);
		}
		test_images.close();
		test_labels.close();
	}
	MNIST()
	{

	}
	void run()
	{
		loadtraindata();
		cout << "MNIST data loaded" << endl;
		graph.load_parameters("mnist4.bin");
		cout << "MNIST model loaded" << endl;
		train();
		graph.save_parameters("mnist4.bin");
		cout << "MNIST model saved" << endl;
		loadtestdata();
		cout << "MNIST test data loaded" << endl;
		int cnt = 0;
		for(int i = 0;i < 10000;++i)cnt += graph.test(test_data[i],test_label[i]);
		cout << cnt << "/10000 correct" << endl;
	}
#pragma pack(2)
	struct BITMAPFILEHEADER /* size: 14 */
	{
		unsigned short bfType;	// 文件的类型，该值必需是0x4D42，也就是字符'BM'。
		unsigned long  bfSize;	// 位图文件的大小，用字节为单位
		unsigned short bfReserved1;// 保留，必须设置为0
		unsigned short bfReserved2;// 保留，必须设置为0
		unsigned long  bfOffBits;// 位图数据距离文件开头偏移量，用字节为单位
	} header;
	struct WINBMPINFOHEADER  /* size: 40 */
	{
		unsigned long  biSize;		// BITMAPINFOHEADER结构所需要的字数
		unsigned long  biWidth;		// 图像宽度，单位为像素
		signed long  biHeight;		// 图像高度，单位为像素，负数，则说明图像是正向的
		unsigned short biPlanes;		// 为目标设备说明位面数，其值将总是被设为1
		unsigned short biBitCount;	// 一个像素占用的bit位，值位1、4、8、16、24、32
		unsigned long  biCompression;// 压缩类型
		unsigned long  biSizeImage;	// 位图数据的大小，以字节为单位
		unsigned long  biXPelsPerMeter;// 水平分辨率，单位 像素/米
		unsigned long  biYPelsPerMeter;// 垂直分辨率，单位 像素/米
		unsigned long  biClrUsed;	// 
		unsigned long  biClrImportant;// 
	} win_header;
	struct RGB
	{
		unsigned char R;
		unsigned char G;
		unsigned char B;
	};
	void load_displaydata(string image_path = "C:\\Users\\pyh\\Desktop\\1.bmp")//处理BMP
	{
		ifstream display_data(image_path,ios::binary | ios::in);
		if(!display_data.is_open())
		{
			cout << "BMP file not found";
			return;
		}
		display_data.read((char*)&header,sizeof(header));
		display_data.read((char*)&win_header,sizeof(win_header));
		/*cout << header.bfType << endl;
		cout << header.bfSize << endl;
		cout << header.bfReserved1 << endl;
		cout << header.bfReserved2 << endl;
		cout << header.bfOffBits << endl;
		cout << "BITMAPINFOHEADER结构信息：" << endl;
		cout << "结构大小: " << win_header.biSize << " 字节" << endl;
		cout << "图像宽度: " << win_header.biWidth << " 像素" << endl;
		cout << "图像高度: " << win_header.biHeight << " 像素" << endl;
		cout << "位面数: " << win_header.biPlanes << endl;
		cout << "每像素位数: " << win_header.biBitCount << " bit" << endl;
		cout << "压缩类型: " << win_header.biCompression << endl;
		cout << "图像数据大小: " << win_header.biSizeImage << " 字节" << endl;
		cout << "水平分辨率: " << win_header.biXPelsPerMeter << " 像素/米" << endl;
		cout << "垂直分辨率: " << win_header.biYPelsPerMeter << " 像素/米" << endl;
		cout << "实际使用颜色数: " << win_header.biClrUsed << endl;
		cout << "重要颜色数: " << win_header.biClrImportant << endl;*/
		if(win_header.biWidth != 28 || win_header.biHeight != 28)
		{
			cout << "文件大小错误 无法进行分类预测" << endl;
			return;
		}
		double arr[28][28];
		for(int i = 27;i >= 0;--i)
		{
			for(int j = 0;j < 28;++j)
			{
				RGB tmpRGB;
				display_data.read((char*)&tmpRGB,sizeof(tmpRGB));
				//cout << (int)tmpRGB.R << " " << (int)tmpRGB.G << " " << (int)tmpRGB.B << " " << endl;
				arr[i][j] = ((((int)tmpRGB.R + (int)tmpRGB.G + (int)tmpRGB.B) / 3.0)) / 255.0;
				//arr[i][j] = arr[i][j] < 0.1 ? 0.1 : 0.8;
			}
		}
		for(int i = 0;i < 28;++i)
		{
			for(int j = 0;j < 28;++j)
			{
				display_image.arr[i * 28 + j] = 1 - arr[i][j];//mnist数据集是黑底白字，我们写的是白底黑字
				//cout << 1 - arr[i][j] << ' ';
			}
			//cout << endl;
		}
	}
	void predict_display_data()
	{
		graph.load_parameters("mnist4.bin");
		cout << "MNIST model loeded" << endl;
		cout << "预测的数字是" << graph.predict(display_image);
	}
	~MNIST()
	{
		if(train_data)delete[] train_data;
		if(train_label)delete[] train_label;
		if(test_data)delete[] test_data;
		if(test_label)delete[] test_label;
	}
};
int Parameter::max_epoch = 1;
int Parameter::batch_size = 100;
double Parameter::lr = 2e-3;
double Parameter::wd1 = 0;
double Parameter::wd2 = 2e-6;
int main()
{
	srand(time(0));
	MNIST mnist;
	mnist.run();
	return 0;
	//batch_size = 300 time 14min  max_epoch =1
}


//1 epoch 8462/10000 correct -> 8807/10000 correct ->8951/10000 correct -> 9014/10000 correct ->9125/10000 correct -> 9162/correct correct
//lr: from 1e-2 -> 4e-3   9165->9176->9193->9210->9221->9230->9234->训练10次->9312->9320 9340 9346 9342 9355
//8989 9173 9203 9217 9225 9233 mnist3
//1132 3050 6878 8067 8454 8649 8802 8898 8969 9051 9055 9062 9076 9093 9102 9114 10次-> 9213 9223 10次-> 9298 9303 9318 9328
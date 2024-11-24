#include <iostream>
#include <cstdlib>
#include <chrono>
#include <fstream>
#include <random>
#include <thread>
#include <atomic>
#include <mutex>
#include <vector>
#include "./hnswlib/hnswlib/hnswlib.h"
#include <numeric>
#include <map>

#define EIGEN_VECTORIZE_SSE4_2
#define EIGEN_VECTORIZE_AVX
#define EIGEN_USE_MKL_ALL
#define EIGEN_USE_BLAS
#include "./eigen-3.4.0/Eigen/Dense"

using Eigen::MatrixXf;
using Eigen::MatrixXd;
using Eigen::VectorXf;
using Eigen::VectorXd;
using Eigen::Matrix;
using Eigen::Map;
using Eigen::aligned_allocator;
using Eigen::FullPivLU;

using namespace std;
using namespace hnswlib;

int ef_construction = 600;
int ef_search = 10;
int M = 40;

int search_k = 10;
int rate = 8;
int efs_max = 10000;
int efs_step = 10;

unsigned dim = 128;
unsigned points_num;
unsigned query_num, query_dim;
unsigned result_num, result_k;

double s = 1024;
double beta_num = 450;


MatrixXd M1(dim / 2 + 4, dim / 2 + 4);
MatrixXd M2(dim / 2 + 4, dim / 2 + 4);
MatrixXd M3(dim * 2 + 16, dim * 2 + 16);
MatrixXd Mup(dim + 8, dim * 2 + 16);
MatrixXd Mdown(dim + 8, dim * 2 + 16);
VectorXd Pai1(dim);
VectorXd Pai2(dim + 8);
double R1, R2, R3, R4;
VectorXd k1(2 * dim + 16);
VectorXd k2(2 * dim + 16);
VectorXd k3(2 * dim + 16);
VectorXd k4(2 * dim + 16);
VectorXd k(2 * dim + 16);

vector<vector<VectorXd>> data_dce(points_num);
vector<VectorXd> query_dce(query_num);

int compare_times=0;


void saveToFile(const std::vector<Eigen::VectorXd>& data, const std::string& filename) {
    std::ofstream outFile(filename,std::ios::binary);
    if (!outFile) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }

    size_t numVectors = data.size();
    outFile.write(reinterpret_cast <const char*> (&numVectors),sizeof (numVectors));

    for (const auto& vec : data) {
        size_t vecSize = vec.size();
        outFile.write(reinterpret_cast <const char*> (&vecSize),sizeof (vecSize));
        for (int i = 0; i < vec.size(); ++i) {
            outFile.write(reinterpret_cast <const char*> (&vec[i]),sizeof (double));
        }
    }

    outFile.close();
}

std::vector<Eigen::VectorXd> loadFromFile(const std::string& filename) {
    std::ifstream inFile(filename,std::ios::binary);
    if (!inFile) {
        std::cerr << "Error opening file for reading: " << filename << std::endl;
        return {};
    }
    std::vector<Eigen::VectorXd> data;
    size_t numVectors;
    inFile.read(reinterpret_cast<char *>(&numVectors),sizeof (numVectors));
    for (size_t i = 0; i < numVectors; ++i) {
        size_t vecSize;
        inFile.read(reinterpret_cast<char *>(&vecSize),sizeof (vecSize));
        Eigen::VectorXd vec(vecSize);
        for (size_t j = 0; j < vecSize; ++j) {
            inFile.read(reinterpret_cast <char*> (&vec[j]),sizeof (double));
        }
        data.push_back(vec);
    }
    inFile.close();
    return data;
}


void load_data_ivecs(basic_string<char> filename,
                     std::vector<std::vector<int> >& results, unsigned &num, unsigned &dim) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }
    in.read((char *)&dim, 4);

    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t)ss;
    num = (unsigned)(fsize / (dim + 1) / 4);
    results.resize(num);
    for (unsigned i = 0; i < num; i++) results[i].resize(dim);
    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num; i++) {
        in.seekg(4, std::ios::cur);
        in.read((char*)results[i].data(), dim * 4);
    }
    in.close();
}

void load_data_fvecs(basic_string<char> filename, float*& data, unsigned& num, unsigned& dim) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }
    in.read((char*)&dim, 4);
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t)ss;
    num = (unsigned)(fsize / (dim + 1) / 4);
    data = new float[(size_t)num * (size_t)dim];

    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num; i++) {
        in.seekg(4, std::ios::cur);
        in.read((char*)(data + i * dim), dim * 4);
    }
    in.close();
}


template<class Function>
inline void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn) {
    if (numThreads <= 0) {
        numThreads = std::thread::hardware_concurrency();
    }

    if (numThreads == 1) {
        for (size_t id = start; id < end; id++) {
            fn(id, 0);
        }
    } else {
        std::vector<std::thread> threads;
        std::atomic<size_t> current(start);

        std::exception_ptr lastException = nullptr;
        std::mutex lastExceptMutex;

        for (size_t threadId = 0; threadId < numThreads; ++threadId) {
            threads.push_back(std::thread([&, threadId] {
                while (true) {
                    size_t id = current.fetch_add(1);

                    if (id >= end) {
                        break;
                    }

                    try {
                        fn(id, threadId);
                    } catch (...) {
                        std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
                        lastException = std::current_exception();
                        current = end;
                        break;
                    }
                }
            }));
        }
        for (auto &thread : threads) {
            thread.join();
        }
        if (lastException) {
            std::rethrow_exception(lastException);
        }
    }
}


void graph_build(float* data_base_source, HierarchicalNSW<float> *hnsw){

    auto *id_label = new int[points_num];
    for(int i = 0; i<points_num;i++){
        id_label[i] = i;
    }
    auto t0 = chrono::high_resolution_clock::now();

    ParallelFor(0, points_num, 40, [&](size_t row, size_t threadId) {
        hnsw->addPoint((void*)(data_base_source + dim * row), id_label[row]);
    });

    auto t1 = chrono::high_resolution_clock::now();
    cout << "build time：" << chrono::duration_cast<chrono::microseconds>(t1-t0).count() << " ms" << endl;

    hnsw->saveIndex("hnsw.bin");

}


vector<vector<hnswlib::labeltype>>
graph_search(float* data_query_source, HierarchicalNSW<float> *hnsw, int k_num, vector<vector<int>> data_result, int finalK){

    std::vector<double> recall;
    hnsw->setEf(ef_search);

    vector<vector<labeltype>> graph_search_result(result_num);

    auto t2 = chrono::high_resolution_clock::now();
    for(int i = 0; i < result_num; i++){
        hnsw->searchKnn(data_query_source+i*dim, k_num);
    }
    auto t3 = chrono::high_resolution_clock::now();

    for(int i = 0; i < result_num; i++){

        vector<labeltype> res = hnsw->searchKnn(data_query_source+i*dim, k_num);

        graph_search_result[i] = res;

        long count = count_if(data_result[i].begin(), data_result[i].begin()+finalK, [&](int x) { return std::find(res.begin(), res.end(), x) != res.end(); });
        recall.push_back((double)count/(double)finalK);
    }

    cout << "graph search = " << k_num << " , ef_search = " << ef_search << endl;
    double recalls = accumulate(recall.begin(), recall.end(), 0.0)/(double)query_num;
    cout << "average recall: " << recalls << endl;

    cout << "search time：" << chrono::duration_cast<chrono::microseconds>(t3-t2).count() << " ms" << endl;

    return graph_search_result;
}

double uniformVector() {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 gen(seed);
    uniform_real_distribution<> dis(0.0, 1.0);
    return dis(gen);
}

VectorXf normalVector(int size) {
    VectorXf vec(size);
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 gen(seed);
    normal_distribution<float> dis(0.0, 1.0);
    for (int i = 0; i < size; ++i) {
        vec(i) = dis(gen);
    }
    return vec;
}


float * encrypt_data(const float * data, const int size){
    Eigen::Map<Eigen::MatrixXf> datavector((float *) data, dim, size);
    for(int i=0; i<datavector.cols(); i++){
        VectorXf u = normalVector((int) dim);
        double x0 = uniformVector();
        double x = s * beta_num * pow(x0, 1.0/(float)dim) / 4.0;
        VectorXf lambda = u * x / u.norm();
        datavector.col(i) = (s * datavector.col(i) + lambda).transpose();
    }
    return datavector.data();
}

void keygen() {

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    M1 = MatrixXd::Random(dim / 2 + 4, dim / 2 + 4);
    M2 = MatrixXd::Random(dim / 2 + 4, dim / 2 + 4);
    M3 = MatrixXd::Random(dim * 2 + 16, dim * 2 + 16);
    Mup = M3.topRows(dim + 8);
    Mdown = M3.bottomRows(dim + 8);

    Pai1 = VectorXd(dim);
    Pai2 = VectorXd(dim+8);
    k1 = VectorXd(2 * dim + 16);
    k2 = VectorXd(2 * dim + 16);
    k3 = VectorXd(2 * dim + 16);
    k4 = VectorXd(2 * dim + 16);
    k = VectorXd(2 * dim + 16);
    for (int i = 0; i < dim; ++i) {
        Pai1(i) = i;
    }
    std::shuffle(Pai1.data(), Pai1.data() + Pai1.size(), gen);

    for (int i = 0; i < dim + 8; ++i) {
        Pai2(i) = i;
    }
    std::shuffle(Pai2.data(), Pai2.data() + Pai2.size(), gen);

    R1 = dis(gen);
    R2 = dis(gen);
    R3 = dis(gen);
    R4 = dis(gen);

    k = VectorXd::Random(2 * dim + 16);
    k1 = VectorXd::Random(2 * dim + 16);
    k2 = VectorXd::Random(2 * dim + 16);
    k3 = k.cwiseQuotient(k1);
    k4 = k.cwiseQuotient(k2);

}

VectorXd aspe_enc(const VectorXd& m) {

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    VectorXd m_temp(dim);
    for (int i = 0; i < dim; i++) {
        m_temp(i) = (i % 2 == 0) ? m(i) + m(i + 1) : m(i - 1) - m(i);
    }
    VectorXd m_temp1(dim);
    for(int i=0; i < dim; i++){
        m_temp1(i) = m_temp((int) Pai1[i]);
    }
    m_temp = m_temp1;

    double rp1 = dis(gen);
    double rp2 = dis(gen);
    double r1 = dis(gen);
    double r2 = dis(gen);
    double r3 = dis(gen);

    VectorXd m1(dim / 2 + 4);
    m1.head(dim / 2) = m_temp.head(dim / 2);
    m1.tail(4) << rp1, -rp1, r1, r2;

    VectorXd m2(dim / 2 + 4);
    m2.head(dim / 2) = m_temp.tail(dim / 2);
    m2.tail(4) << rp2, rp2, r3, (-m.squaredNorm() - r1 * R1 - r2 * R2 - r3 * R3) / R4;

    VectorXd newm_temp(dim + 8);
    newm_temp.head(dim / 2 + 4) = M1.transpose() * m1;
    newm_temp.tail(dim / 2 + 4) = M2.transpose() * m2;

    VectorXd newm(dim + 8);
    for(int i=0; i<dim+8; i++){
        newm(i) = newm_temp((int) Pai2[i]);
    }
    return newm;
}

vector<VectorXd> dce_enc(const VectorXd& m) {

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    double rup = dis(gen);
    double rdown = dis(gen);

    vector<VectorXd> enc(4);

    enc[0] = rup * ((m.transpose() * Mup).transpose() + VectorXd::Ones(2 * dim + 16)).cwiseQuotient(k1);
    enc[1] = rup * ((m.transpose() * Mup).transpose() - VectorXd::Ones(2 * dim + 16)).cwiseQuotient(k2);
    enc[2] = rdown * ((m.transpose() * Mdown).transpose() + VectorXd::Ones(2 * dim + 16)).cwiseQuotient(k3);
    enc[3] = rdown * ((m.transpose() * Mdown).transpose() - VectorXd::Ones(2 * dim + 16)).cwiseQuotient(k4);

    return enc;
}

VectorXd aspe_trapdoor(const VectorXd& q) {

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    VectorXd q_temp(dim);
    for (int i = 0; i < dim; i++) {
        q_temp(i) = (i % 2 == 0) ? q(i) + q(i + 1) : q(i - 1) - q(i);
    }
    VectorXd q_temp1(dim);
    for(int i=0; i < dim; i++){
        q_temp1(i) = q_temp((int) Pai1[i]);
    }
    q_temp = q_temp1;

    double rq1 = dis(gen);
    double rq2 = dis(gen);

    VectorXd q1(dim / 2 + 4);
    q1.head(dim / 2) = q_temp.head(dim / 2);
    q1.tail(4) << rq1, rq1, R1, R2;

    VectorXd q2(dim / 2 + 4);
    q2.head(dim / 2) = q_temp.tail(dim / 2);
    q2.tail(4) << rq2, -rq2, R3, R4;

    VectorXd trapdoor_temp(dim + 8);
    trapdoor_temp.head(dim / 2 + 4) = (M1.completeOrthogonalDecomposition().pseudoInverse() * q1).transpose();
    trapdoor_temp.tail(dim / 2 + 4) = (M2.completeOrthogonalDecomposition().pseudoInverse() * q2).transpose();

    VectorXd trapdoor(dim + 8);
    for(int i=0; i<dim+8; i++){
        trapdoor(i) = -trapdoor_temp((int) Pai2[i]);
    }

    return trapdoor;
}

VectorXd dce_trapdoor(const VectorXd& q) {

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    double rq = dis(gen);

    VectorXd q_double(2*dim + 16);
    q_double.head(dim + 8) = q;
    q_double.tail(dim + 8) = -q;
    VectorXd trapdoor = rq * (M3.completeOrthogonalDecomposition().pseudoInverse() * q_double).cwiseProduct(k);

    return trapdoor;
}

bool dce_check(const vector<VectorXd>& a, const vector<VectorXd>& b, const VectorXd& c){
    compare_times+=1;
    return (a[0].cwiseProduct(b[2]) - a[1].cwiseProduct(b[3])).dot(c) <= 0;
}


void siftdown(vector<int>& heap, int startpos, int pos, const VectorXd& trapdoor) {
    int newitem = heap[pos];
    while (pos > startpos) {
        int parentpos = (pos - 1) >> 1;
        int parent = heap[parentpos];
        if (dce_check(data_dce[newitem], data_dce[parent], trapdoor)==0){
            heap[pos] = parent;
            pos = parentpos;
            continue;
        }
        break;
    }
    heap[pos] = newitem;
}

void siftup(vector<int>& heap, int pos, const VectorXd& trapdoor) {
    int endpos = heap.size();
    int startpos = pos;
    int newitem = heap[pos];
    int childpos = 2 * pos + 1;
    while (childpos < endpos) {
        int rightpos = childpos + 1;
        if (rightpos < endpos && dce_check(data_dce[heap[childpos]], data_dce[heap[rightpos]], trapdoor))
            childpos = rightpos;
        heap[pos] = heap[childpos];
        pos = childpos;
        childpos = 2 * pos + 1;
    }
    heap[pos] = newitem;
    siftdown(heap, startpos, pos, trapdoor);
}


void heapify(vector<int>& x, const VectorXd& trapdoor) {
    int n = x.size();
    for (int i = n / 2 - 1; i >= 0; i--)
        siftup(x, i, trapdoor);
}

int heappushpop(vector<int>& heap, int item, const VectorXd& trapdoor) {
    if (!heap.empty() && dce_check(data_dce[item], data_dce[heap[0]], trapdoor)) {
        item = exchange(heap[0], item);
        siftup(heap, 0, trapdoor);
    }
    return item;
}

void heap_k(vector<int>& data0, std::vector<int>& data1, const VectorXd& trapdoor) {
    heapify(data0, trapdoor);
    for (int num : data1){
        heappushpop(data0, num, trapdoor);
    }
}



void second_search(vector<vector<labeltype>> graph_search_result, vector<vector<int>> data_result, int finalK, float* database_new){
    vector<double> recall0, recall1, recall2;
    long sum_time0 = 0;

    if (graph_search_result[0].size()<=finalK){
        cout << "time0 = " << sum_time0 << " microseconds" << endl;
        cout << "average recall0: 0" << endl;
        return;
    }
    for(int i=0; i<query_num; i++){

        vector<int> result_id0(graph_search_result[i].size()), result_id1;
        std::transform(graph_search_result[i].begin(), graph_search_result[i].end(), result_id0.begin(), [](unsigned long ul) {
            return static_cast<int>(ul);
        });
        result_id1 = result_id0;
        reverse(result_id1.begin(), result_id1.end());

        for(int ki=0;ki<result_id0.size();ki++) {

            VectorXd datai(dim);
            for(int dii=0; dii<dim; dii++){
                datai[dii] = (double)database_new[result_id0[ki]*dim+dii];
            }
            auto res = dce_enc(aspe_enc(datai));

            data_dce[result_id0[ki]] = res;
        }

        auto t0 = chrono::high_resolution_clock::now();
        vector<int> data_id1(result_id0.begin(), result_id0.end()-finalK);
        vector<int> data_id0(result_id0.end()-finalK, result_id0.end());
        heap_k(data_id0, data_id1, query_dce[i]);
        auto t1 = chrono::high_resolution_clock::now();
        sum_time0 += chrono::duration_cast<chrono::microseconds>(t1-t0).count();

        long count = count_if(data_result[i].begin(), data_result[i].begin()+finalK, [&](int x) { return std::find(data_id0.begin(), data_id0.end(), x) != data_id0.end(); });
        recall0.push_back((double)count/(double)data_id0.size());

        data_dce.clear();
        data_dce.shrink_to_fit();
        data_dce = vector<vector<VectorXd>>(points_num);
    }

    cout << "time0 = " << sum_time0 << " microseconds" << endl;

    double recalls0 = accumulate(recall0.begin(), recall0.end(), 0.0)/(double)query_num;
    cout << "average recall0: " << recalls0 << endl;

}


void ppanns_exp(const char *database_path, string dataquery_path, string groundtruth_path){

    float* database;
    float* dataquery;
    vector<vector<int>> data_result;

    load_data_fvecs(database_path, database, points_num, dim);
    load_data_fvecs(dataquery_path, dataquery, query_num, query_dim);
    load_data_ivecs(groundtruth_path, data_result, result_num, result_k);

    keygen();
    cout << "data load over and key has been generated" << endl;
    data_dce = vector<vector<VectorXd>>(points_num);
    query_dce = vector<VectorXd>(query_num);



    double* query_double;
    query_double = static_cast<double *>(malloc(query_num * dim * sizeof(double)));
    ParallelFor(0, query_num * dim, 40, [&](size_t i, size_t threadId) {
        query_double[i] = (double)dataquery[i];
    });
    Eigen::Map<Eigen::MatrixXd> query_vector(query_double, dim, query_num);

    auto t2 = chrono::high_resolution_clock::now();
    ParallelFor(0,  query_num, 40, [&](size_t i, size_t threadId) {
        query_dce[i] = dce_trapdoor(aspe_trapdoor(query_vector.col(i)));
    });
    auto t3 = chrono::high_resolution_clock::now();
    cout << "DCE query encrypt time = " << chrono::duration_cast<chrono::microseconds>(t3-t2).count() << endl;

    delete query_double;

    auto et0 = chrono::high_resolution_clock::now();
    database = encrypt_data(database, (const int) points_num);
    auto et1 = chrono::high_resolution_clock::now();
    auto et2 = chrono::high_resolution_clock::now();
    dataquery = encrypt_data(dataquery, (const int) query_num);
    auto et3 = chrono::high_resolution_clock::now();

    cout << "DCPE encrypt data time = " << chrono::duration_cast<chrono::microseconds>(et1-et0).count() << " ms" << endl;
    cout << "DCPE encrypt query time = " << chrono::duration_cast<chrono::microseconds>(et3-et2).count() << " ms" << endl;

    cout << "Encrypting over and clear over!" << endl;

    L2Space l2space(dim);
    HierarchicalNSW<float> *hnsw;
    hnsw = new HierarchicalNSW<float>(&l2space, points_num, M, ef_construction);
    graph_build(database, hnsw);
    delete[] database;

    float* database_new;
    load_data_fvecs(database_path, database_new, points_num, dim);

    for(int efs = ef_search; efs <= efs_max; efs += efs_step){
        ef_search = efs;
        auto graph_search_result = graph_search(dataquery, hnsw, ef_search, data_result, search_k);
        second_search(graph_search_result, data_result, search_k, database_new);
        cout << "compare_times = " << compare_times << endl;
        compare_times=0;
    }

    delete hnsw;
    delete[] dataquery;
    data_dce.clear();
    query_dce.clear();
    data_dce.shrink_to_fit();
    query_dce.shrink_to_fit();

}




int main(int argc, char *argv[]){

    M = 40;
    ef_search = 10;
    ef_construction = 600;
    s = 1024.0;
    beta_num = 0.0;
    search_k = 10;
    int exp = 1;
    std::string database_path;
    std::string dataquery_path;
    std::string groundtruth_path;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            std::cout << "Usage: " << argv[0] << " [options]\n";
            std::cout << "Options:\n";
            std::cout << "  -h, --help             Print help messages\n";
            std::cout << "  --M <value>            M (default 40)\n";
            std::cout << "  --efc <value>          efConstruction (default 600)\n";
            std::cout << "  --efs <value>          efSearch (default 10)\n";
            std::cout << "  --efsm <value>         maximum efSearch (default 10000)\n";
            std::cout << "  --efss <value>         efSearch step length (default 10)\n";
            std::cout << "  --s <value>            s (default 1024)\n";
            std::cout << "  --beta <value>         beta (default 0)\n";
            std::cout << "  --k <value>            k (default 10)\n";
            std::cout << "  --ratio <value>        ratio_k  (default 8)\n";
            std::cout << "  --exp <value>          exp (default 1)\n";
            std::cout << "  --database <path>      Database path (required)\n";
            std::cout << "  --dataquery <path>     Data query path (required)\n";
            std::cout << "  --groundtruth <path>   Ground truth path (required)\n";
            return 0;
        } else if (std::strcmp(argv[i], "--M") == 0) {
            if (i + 1 < argc) M = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--efc") == 0) {
            if (i + 1 < argc) ef_construction = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--efs") == 0) {
            if (i + 1 < argc) ef_search = std::atoi(argv[++i]);
        }else if (std::strcmp(argv[i], "--efsm") == 0) {
            if (i + 1 < argc) efs_max = std::atoi(argv[++i]);
        }else if (std::strcmp(argv[i], "--efss") == 0) {
            if (i + 1 < argc) efs_step = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--s") == 0) {
            if (i + 1 < argc) s = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--beta") == 0) {
            if (i + 1 < argc) beta_num = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--k") == 0) {
            if (i + 1 < argc) search_k = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--ratio") == 0) {
            if (i + 1 < argc) rate = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--exp") == 0) {
            if (i + 1 < argc) exp = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--database") == 0) {
            if (i + 1 < argc) database_path = argv[++i];
        } else if (std::strcmp(argv[i], "--dataquery") == 0) {
            if (i + 1 < argc) dataquery_path = argv[++i];
        } else if (std::strcmp(argv[i], "--groundtruth") == 0) {
            if (i + 1 < argc) groundtruth_path = argv[++i];
        } else {
            std::cerr << "Unknown option: " << argv[i] << std::endl;
            return 1;
        }
    }

    if (database_path.empty() || dataquery_path.empty() || groundtruth_path.empty()) {
        std::cerr << "Error: Missing required arguments.\n";
        return 1;
    }

    std::cout << "M: " << M << std::endl;
    std::cout << "efConstruction: " << ef_construction << std::endl;
    std::cout << "efSearch: " << ef_search << std::endl;
    std::cout << "maximum efSearch: " << efs_max << std::endl;
    std::cout << "efSearch step length: " << efs_step << std::endl;
    std::cout << "s: " << s << std::endl;
    std::cout << "Beta: " << beta_num << std::endl;
    std::cout << "k: " << search_k << std::endl;
    std::cout << "ratio: " << rate << std::endl;
    std::cout << "exp: " << exp << std::endl;
    std::cout << "Database: " << database_path << std::endl;
    std::cout << "Data query: " << dataquery_path << std::endl;
    std::cout << "Ground truth: " << groundtruth_path << std::endl;

    Eigen::initParallel();

    ppanns_exp(database_path.data(), dataquery_path.data(), groundtruth_path.data());

    return 0;
}




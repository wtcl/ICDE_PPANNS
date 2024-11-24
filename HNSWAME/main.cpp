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
int efs_step = 100;

unsigned dim;
unsigned points_num;
unsigned query_num, query_dim;
unsigned result_num, result_k;

double s = 1024;
double beta_num = 450;


vector<MatrixXd> sk(32);
vector<MatrixXd> sk_inv(32);
vector<vector<MatrixXd>> ame_data(points_num);
Eigen::MatrixXd data_vector;
Eigen::MatrixXd query_vector;


int compare_times=0;

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

void initialize() {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dis(0.0, 1.0);

    for (int i = 0; i < 32; ++i) {
        sk[i] = MatrixXd::Random(2*dim+6, 2*dim+6);
        sk_inv[i] = sk[i].completeOrthogonalDecomposition().pseudoInverse();
    }
}

vector<MatrixXd> ameDBEnc(VectorXd x) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dis(0.0, 100.0);

    double xNorm = -0.5 * x.norm() * x.norm();
    x.conservativeResize(x.size() + 1);
    x.tail(1) << xNorm;

    VectorXd xl = VectorXd::Zero(2*dim+6);
    VectorXd xr = VectorXd::Zero(2*dim+6);

    for(int i=0; i<dim+1; i++){
        xl(i) = x(i);
        xl(i + dim + 1) = -1;
        xr(i) = 1;
        xr(i + dim + 1) = x(i);
    }

    vector<double> r(3);
    for (int i = 0; i < 3; ++i) {
        r[i] = dis(gen);
    }
    sort(r.begin(), r.end());

//    VectorXd xl_caret = (r[2] * xl).eval();
//    xl_caret.tail(4) << r[1], r[1], r[0], 1.0;
    VectorXd xl_caret = (xl).eval();
    xl_caret.tail(4) << 0.0, 0.0, 0.0, 0.0;

    for (int i = 0; i < 3; ++i) {
        r[i] = dis(gen);
    }
    sort(r.begin(), r.end());
//    VectorXd xl_tilde = (r[2] * xl).eval();
//    xl_tilde.tail(4) << r[1], r[1], r[0], 1.0;
    VectorXd xl_tilde = (xl).eval();
    xl_tilde.tail(4) << 0.0, 0.0, 0.0, 0.0;

    for (int i = 0; i < 3; ++i) {
        r[i] = dis(gen);
    }
    sort(r.begin(), r.end());
//    VectorXd xr_caret = (r[2] * xr).eval();
//    xr_caret.tail(4) << r[1], -r[1], r[0], 1.0;
    VectorXd xr_caret = (xr).eval();
    xr_caret.tail(4) << 0.0, 0.0, 0.0, 0.0;

    for (int i = 0; i < 3; ++i) {
        r[i] = dis(gen);
    }
    sort(r.begin(), r.end());
//    VectorXd xr_tilde = (r[2] * xr).eval();
//    xr_tilde.tail(4) << r[1], -r[1], r[0], 1.0;
    VectorXd xr_tilde = (xr).eval();
    xr_tilde.tail(4) << 0.0, 0.0, 0.0, 0.0;

    VectorXd xl_caret_0 = VectorXd::Random(xl_caret.size());
    VectorXd xl_caret_1 = xl_caret - xl_caret_0;
    VectorXd xl_tilde_0 = VectorXd::Random(xl_tilde.size());
    VectorXd xl_tilde_1 = xl_tilde - xl_tilde_0;

    VectorXd xr_caret_0 = VectorXd::Random(xr_caret.size());
    VectorXd xr_caret_1 = xr_caret - xr_caret_0;
    VectorXd xr_tilde_0 = VectorXd::Random(xr_tilde.size());
    VectorXd xr_tilde_1 = xr_tilde - xr_tilde_0;

    vector<MatrixXd> Enc0(8);
    vector<MatrixXd> Enc1(8);
    vector<MatrixXd> Enc2(8);
    vector<MatrixXd> Enc3(8);

    for(int i=0; i<4; i++){
        Enc0[i] = (xl_caret_0.transpose() * sk[i]).transpose();
        Enc1[i] = sk_inv[i+8] * xr_caret_0;
        Enc2[i] = (xl_tilde_0.transpose() * sk[i+16]).transpose();
        Enc3[i] = sk_inv[i+24] * xr_tilde_0;

        Enc0[i+4] = (xl_caret_1.transpose() * sk[i+4]).transpose();
        Enc1[i+4] = sk_inv[i+12] * xr_caret_1;
        Enc2[i+4] = (xl_tilde_1.transpose() * sk[i+20]).transpose();
        Enc3[i+4] = sk_inv[i+28] * xr_tilde_1;
    }

    vector<MatrixXd> Enc;
    Enc.insert(Enc.end(), Enc0.begin(), Enc0.end());
    Enc.insert(Enc.end(), Enc2.begin(), Enc2.end());
    Enc.insert(Enc.end(), Enc1[0]);
    Enc.insert(Enc.end(), Enc1[4]);
    Enc.insert(Enc.end(), Enc1[1]);
    Enc.insert(Enc.end(), Enc1[5]);
    Enc.insert(Enc.end(), Enc1[2]);
    Enc.insert(Enc.end(), Enc1[6]);
    Enc.insert(Enc.end(), Enc1[3]);
    Enc.insert(Enc.end(), Enc1[7]);
    Enc.insert(Enc.end(), Enc3[0]);
    Enc.insert(Enc.end(), Enc3[4]);
    Enc.insert(Enc.end(), Enc3[1]);
    Enc.insert(Enc.end(), Enc3[5]);
    Enc.insert(Enc.end(), Enc3[2]);
    Enc.insert(Enc.end(), Enc3[6]);
    Enc.insert(Enc.end(), Enc3[3]);
    Enc.insert(Enc.end(), Enc3[7]);

    return Enc;
}

vector<MatrixXd> ameQueryEnc(VectorXd q) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dis(0.0, 100.0);

    q.conservativeResize(q.size() + 1);
    q.tail(1) << 1.0;

    VectorXd q2 = VectorXd::Zero(2 * dim + 6);
    for(int i=0; i<dim+1; i++){
        q2(i) = q(i);
        q2(i+dim+1) = q(i);
    }

    vector<double> r(3);
    for (int i = 0; i < 3; ++i) {
        r[i] = dis(gen);
    }
    sort(r.begin(), r.end());
    double rq = dis(gen);

    MatrixXd Q_caret = MatrixXd::Zero(2*dim+6, 2*dim+6);
//    Q_caret.diagonal().head(q2.size()) = r[2] * q2;
//    Q_caret.diagonal().tail(4) << r[1], r[1], r[0], rq;
    Q_caret.diagonal().head(q2.size()) = q2;
    Q_caret.diagonal().tail(4) << 0.0, 0.0, 0.0, 0.0;

    for (int i = 0; i < 3; ++i) {
        r[i] = dis(gen);
    }
    sort(r.begin(), r.end());

    MatrixXd Q_tilde = MatrixXd::Zero(2*dim+6, 2*dim+6);
//    Q_tilde.diagonal().head(q2.size()) = r[2] * q2;
//    Q_tilde.diagonal().tail(4) << r[1], r[1], r[0], -rq;
    Q_tilde.diagonal().head(q2.size()) = q2;
    Q_tilde.diagonal().tail(4) << 0.0, 0.0, 0.0, 0.0;

    MatrixXd Q_caret_0 = MatrixXd::Random(Q_caret.rows(), Q_caret.cols());
    MatrixXd Q_caret_1 = Q_caret - Q_caret_0;
    MatrixXd Q_tilde_0 = MatrixXd::Random(Q_tilde.rows(), Q_tilde.cols());
    MatrixXd Q_tilde_1 = Q_tilde - Q_tilde_0;

    vector<MatrixXd> Encq(16);

    Encq[0] = sk_inv[0] * Q_caret_0 * sk[8];
    Encq[1] = sk_inv[1] * Q_caret_0 * sk[12];
    Encq[2] = sk_inv[2] * Q_caret_1 * sk[9];
    Encq[3] = sk_inv[3] * Q_caret_1 * sk[13];
    Encq[4] = sk_inv[4] * Q_caret_0 * sk[10];
    Encq[5] = sk_inv[5] * Q_caret_0 * sk[14];
    Encq[6] = sk_inv[6] * Q_caret_1 * sk[11];
    Encq[7] = sk_inv[7] * Q_caret_1 * sk[15];

    Encq[8] = sk_inv[16] * Q_tilde_0 * sk[24];
    Encq[9] = sk_inv[17] * Q_tilde_0 * sk[28];
    Encq[10] = sk_inv[18] * Q_tilde_1 * sk[25];
    Encq[11] = sk_inv[19] * Q_tilde_1 * sk[29];
    Encq[12] = sk_inv[20] * Q_tilde_0 * sk[26];
    Encq[13] = sk_inv[21] * Q_tilde_0 * sk[30];
    Encq[14] = sk_inv[22] * Q_tilde_1 * sk[27];
    Encq[15] = sk_inv[23] * Q_tilde_1 * sk[31];

    return Encq;
}

bool check_ame(const vector<MatrixXd>& Ea, const vector<MatrixXd>& Eb, const vector<MatrixXd>& Eq) {
    compare_times += 1;
    double flag = 0;

    for (int i = 0; i < 8; ++i) {
        flag += (Ea[i].transpose() * Eq[i] * Eb[i+16])(0);
    }
    for (int i = 8; i < 16; ++i) {
        flag += (Ea[i].transpose() * Eq[i] * Eb[i+16])(0);
    }
    return flag >= 0;
}


void siftdown(std::vector<int>& heap, int startpos, int pos, const vector<MatrixXd> trapdoor) {
    int newitem = heap[pos];
    while (pos > startpos) {
        int parentpos = (pos - 1) >> 1;
        int parent = heap[parentpos];
        if (check_ame(ame_data[newitem], ame_data[parent], trapdoor)==0){
            heap[pos] = parent;
            pos = parentpos;
            continue;
        }
        break;
    }
    heap[pos] = newitem;
}

void siftup(std::vector<int>& heap, int pos, const vector<MatrixXd> trapdoor) {
    int endpos = heap.size();
    int startpos = pos;
    int newitem = heap[pos];
    int childpos = 2 * pos + 1;
    while (childpos < endpos) {
        int rightpos = childpos + 1;
        if (rightpos < endpos && check_ame(ame_data[heap[childpos]], ame_data[heap[rightpos]], trapdoor))
            childpos = rightpos;
        heap[pos] = heap[childpos];
        pos = childpos;
        childpos = 2 * pos + 1;
    }
    heap[pos] = newitem;
    siftdown(heap, startpos, pos, trapdoor);
}


void heapify(std::vector<int>& x, const vector<MatrixXd> trapdoor) {
    int n = x.size();
    for (int i = n / 2 - 1; i >= 0; i--)
        siftup(x, i, trapdoor);
}

int heappushpop(vector<int>& heap, int item, const vector<MatrixXd> trapdoor) {
    if (!heap.empty() && check_ame(ame_data[item], ame_data[heap[0]], trapdoor)) {
        item = exchange(heap[0], item);
        siftup(heap, 0, trapdoor);
    }
    return item;
}

void heap_k(std::vector<int>& data0, std::vector<int>& data1, const vector<MatrixXd> trapdoor) {
    heapify(data0, trapdoor);
    for (int num : data1){
        heappushpop(data0, num, trapdoor);
    }
}


void refine(vector<vector<int>> data_result, vector<vector<labeltype>> results, int finalK){
    cout << "second search start: " << endl;
    vector<vector<int>> final_result;
    long sum_second_time = 0;
    compare_times = 0;
    ame_data = vector<vector<MatrixXd>>(points_num);

    for(int i=0; i<query_num; i++){
        ParallelFor(0, results[i].size(), 40, [&](size_t j, size_t threadId) {
            ame_data[results[i][j]]=(ameDBEnc(data_vector.col(results[i][j])));
        });
        auto trapdoor=(ameQueryEnc(query_vector.col(i)));

        auto t2 = chrono::high_resolution_clock::now();
        vector<int> res_id0(results[i].begin(), results[i].end()-finalK);
        vector<int> res_id1(results[i].end()-finalK, results[i].end());
        heap_k(res_id1, res_id0, trapdoor);
        auto t3 = chrono::high_resolution_clock::now();
        final_result.push_back(res_id1);
        sum_second_time += chrono::duration_cast<chrono::microseconds>(t3-t2).count();

        ame_data.clear();
        ame_data.resize(points_num);
    }

    cout << "second search time: " << sum_second_time << endl;

    vector<double> recalls2;
    for(int i=0; i<final_result.size(); i++){
        long count = count_if(data_result[i].begin(), data_result[i].begin()+finalK, [&](int x) { return std::find(final_result[i].begin(), final_result[i].end(), x) != final_result[i].end(); });
        recalls2.push_back((double)count/(double)finalK);
    }
    cout << "recall 2: " << accumulate(recalls2.begin(), recalls2.end(), 0.0) / (double)query_num << endl;

    cout << "compare_times = " << compare_times << endl;
    compare_times = 0;
}





void ppanns_exp(const char *database_path, string dataquery_path, string groundtruth_path){

    float* database;
    float* dataquery;
    vector<vector<int>> data_result;

    load_data_fvecs(database_path, database, points_num, dim);
    load_data_fvecs(dataquery_path, dataquery, query_num, query_dim);
    load_data_ivecs(groundtruth_path, data_result, result_num, result_k);

    cout << "data load over and key has been generated" << endl;

    double* database_double;
    database_double = static_cast<double *>(malloc(points_num * dim * sizeof(double)));
    ParallelFor(0, points_num * dim, 40, [&](size_t i, size_t threadId) {
        database_double[i] = (double)database[i];
    });

    double* query_double;
    query_double = static_cast<double *>(malloc(query_num * dim * sizeof(double)));
    ParallelFor(0, query_num * dim, 40, [&](size_t i, size_t threadId) {
        query_double[i] = (double)dataquery[i];
    });

    Eigen::Map<Eigen::MatrixXd> data_vector(database_double, dim, points_num);
    Eigen::Map<Eigen::MatrixXd> query_vector(query_double, dim, query_num);


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

    for(int efs = rate*search_k; efs <= efs_max; efs += efs_step){
        ef_search = efs;
        auto graph_search_result = graph_search(dataquery, hnsw, rate*search_k, data_result, search_k);
        if(efs!=search_k){
            refine(data_result, graph_search_result, search_k);
        }else{
            cout << "second search start: " << endl;
            cout << "second search time: " << 0 << endl;
            cout << "recall 2: " << 0 << endl;
            cout << "compare_times = 0" << endl;
        }
        compare_times=0;
    }

    delete hnsw;
    delete database;
    delete dataquery;
    ame_data.clear();
    ame_data.shrink_to_fit();

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
        } else if (std::strcmp(argv[i], "--dim")==0){
            if (i + 1 < argc) dim = std::atof(argv[++i]);
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
    std::cout << "Dim: " << dim << std::endl;

    initialize();
    Eigen::initParallel();
    ppanns_exp(database_path.data(), dataquery_path.data(), groundtruth_path.data());


    return 0;
}




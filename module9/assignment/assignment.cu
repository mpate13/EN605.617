#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cusparse.h>

#define CHECK(Stat) { if (Stat != 0) { printf("Error at %d\n", __LINE__); } }

template<typename T>
void load_bin(const std::string& name, std::vector<T>& v) {
    std::ifstream f(name, std::ios::binary | std::ios::ate);
    std::streamsize sz = f.tellg();
    f.seekg(0, std::ios::beg);
    v.resize(sz / sizeof(T));
    f.read(reinterpret_cast<char*>(v.data()), sz);
}

void encode(cudnnHandle_t h, float* d_v, int len) {
    cudnnTensorDescriptor_t desc;
    cudnnActivationDescriptor_t act;
    float a = 1.0f, b = 0.0f;
    cudnnCreateTensorDescriptor(&desc);
    cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, len);
    cudnnCreateActivationDescriptor(&act);
    cudnnSetActivationDescriptor(act, CUDNN_ACTIVATION_SIGMOID, CUDNN_NOT_PROPAGATE_NAN, 0.0f);
    cudnnActivationForward(h, act, &a, desc, d_v, &b, desc, d_v);
}

int main() {
    std::vector<int> h_r, h_c; std::vector<float> h_val;
    load_bin("row_ptr.bin", h_r); load_bin("col_ind.bin", h_c); load_bin("values.bin", h_val);

    int n_books = h_r.size() - 1, n_genres = 1000, nz = h_val.size();
    std::vector<float> h_query(n_genres, 0.0f);
    
    std::cout << "Enter Genre Index (from genre_map.txt) and Weight (1-5). -1 to start:\n";
    int idx; float w;
    while (std::cin >> idx && idx != -1) { std::cin >> w; h_query[idx] = w; }

    float *d_v, *d_out, *d_val; int *d_r, *d_c;
    cudaMalloc(&d_v, n_genres * sizeof(float));
    cudaMalloc(&d_out, n_books * sizeof(float));
    cudaMalloc(&d_r, h_r.size() * sizeof(int));
    cudaMalloc(&d_c, h_c.size() * sizeof(int));
    cudaMalloc(&d_val, h_val.size() * sizeof(float));

    cudaMemcpy(d_v, h_query.data(), n_genres * sizeof(float), cudaMemcpyH2D);
    cudaMemcpy(d_r, h_r.data(), h_r.size() * sizeof(int), cudaMemcpyH2D);
    cudaMemcpy(d_c, h_c.data(), h_c.size() * sizeof(int), cudaMemcpyH2D);
    cudaMemcpy(d_val, h_val.data(), h_val.size() * sizeof(float), cudaMemcpyH2D);

    cudnnHandle_t dnn; cusparseHandle_t sp;
    cudnnCreate(&dnn); cusparseCreate(&sp);
    encode(dnn, d_v, n_genres);
    
    cusparseMatDescr_t desc; cusparseCreateMatDescr(&desc);
    float alpha = 1.0f, beta = 0.0f;
    cusparseScsrmv(sp, CUSPARSE_OPERATION_NON_TRANSPOSE, n_books, n_genres, nz, 
                   &alpha, desc, d_val, d_r, d_c, d_v, &beta, d_out);

    std::vector<float> h_scores(n_books);
    cudaMemcpy(h_scores.data(), d_out, n_books * sizeof(float), cudaMemcpyD2H);

    // Sorting and Title Mapping
    std::vector<std::pair<float, int>> results(n_books);
    for(int i=0; i<n_books; ++i) results[i] = {h_scores[i], i};
    std::sort(results.begin(), results.end(), [](auto& a, auto& b){ return a.first > b.first; });

    std::vector<std::string> titles; std::ifstream t_f("titles.txt");
    std::string line; while(std::getline(t_f, line)) titles.push_back(line);

    std::cout << "\n--- TOP 10 BOOKS ---\n";
    for(int i=0; i<10; ++i) std::cout << titles[results[i].second] << "\n";

    return 0;
}
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <math.h>
#include "fwg.h"
#include <stdio.h>
#include <stdlib.h>
#include <float.h>

#include <iostream>
#include <thread>
#include <algorithm>
#include <vector>
#include <array>
#include <future>
#include <utility>

#define min(a,b) (a<=b?a:b)
#define max(a,b) (a>=b?a:b)

void print_vec(std::vector<double> v) {
    for(int i=0;i<v.size();++i) {
        printf("%.2f ", v[i]);
    }
    printf("\n");
}

void print_emb(embedding_t v) {
    for(int i=0;i<v.size();++i) {
        printf("(%.2f, %.2f) -- ", v[i].first, v[i].second);
    }
    printf("\n");
}

double sliced_wasserstein_distance(
    embedding_t embedding_i,
    embedding_t embedding_j,
    int M
) {

    // printf("\n###############\nInput diagrams: \n");
    // print_emb(embedding_i);
    // print_emb(embedding_j);

    int size_i = embedding_i.size();
    int size_j = embedding_j.size();

    int k, l;
    int u = size_i + size_j;

    embedding_t vec1(u);
    embedding_t vec2(u);

    for (k=0; k<size_i; k++) {
        double birth = embedding_i[k].first;
        double death = embedding_i[k].second;

        vec1[k] = {birth, death};
        double mean =  (birth+death)/2.0;
        vec2[k] = {mean, mean};
    }

    for (k=0; k<size_j; k++) {
        double birth = embedding_j[k].first;
        double death = embedding_j[k].second;

        vec2[size_i+k] = {birth, death};
        double mean =  (birth+death)/2.0;
        vec1[size_i+k] = {mean, mean};
    }
    

    double sw = 0;
    double theta = - M_PI / 2.0;
    double s = M_PI / M;

    // printf("Enriched diagrams: \n");
    // print_emb(vec1);
    // print_emb(vec2);

    for (k=0; k<M; k++) {

        // printf("\nK=%i (cos(theta)=%f, sin(theta)=%f)\n", k, cos(theta), sin(theta));
    
        std::vector<double> v1(u);
        std::vector<double> v2(u);
        for (l=0; l<u; l++) {
            v1[l] = vec1[l].first * cos(theta) + vec1[l].second * sin(theta);
            v2[l] = vec2[l].first * cos(theta) + vec2[l].second * sin(theta);

            if (isnan(v1[l])) v1[l]=0;
            if (isnan(v2[l])) v2[l]=0;
        }

        
        // printf("v1 (before sort)=");
        // print_vec(v1);
        // printf("v2 (before sort)=");
        // print_vec(v2);

        std::sort(v1.begin(), v1.end());
        std::sort(v2.begin(), v2.end());

        // printf("v1=");
        // print_vec(v1);
        // printf("v2=");
        // print_vec(v2);

        double norm1 = 0.0;
        for (l=0; l<u; l++) {
            double raw_val = v1[l] - v2[l];
            // printf("%f - %f = %f\n", v1[l], v2[l], raw_val);
            if (isinf(raw_val)) {
                // If at least one positive or negative infinity is encountered, the whole
                // distance will be infinity so we can return immediately
                // printf("Encountered inf after diff, return inf for distance\n");
                return std::numeric_limits<double>::infinity();
            }
            else if (!isnan(raw_val)) {
                norm1 += fabs(raw_val);
            } 
            // printf("norm1 = %f\n", norm1);
        }

        sw = sw + s * norm1;
        theta = theta + s;
    }
    
    return (1 / M_PI) * sw;

}

PyObject* fast_wasserstein_distances(
    std::vector<embedding_t> embeddings_in,
    std::vector<embedding_t> embeddings_out,
    int M
)
 {
    int n = embeddings_in.size();
    int m = embeddings_out.size();

    PyObject* gram = PyList_New(n*m);

    std::vector<std::future<double>> my_futures;

    for (int i=0; i<n; ++i) {
        embedding_t embedding_i = embeddings_in[i];

        my_futures.clear();

        for (int j=0; j<m; ++j) {
            embedding_t embedding_j = embeddings_out[j];

            my_futures.push_back(std::async(std::launch::async, sliced_wasserstein_distance, embedding_i,
                embedding_j,
                M));
        }

        for (int j=0; j<m; ++j) {
            PyList_SET_ITEM(gram, i*m+j, PyFloat_FromDouble(my_futures[j].get()));
        }

    }

    import_array();
    PyArrayObject* gramNumpy = (PyArrayObject*) PyArray_FromAny(gram, PyArray_DescrFromType(NPY_FLOAT64), 1, 1, NPY_ARRAY_DEFAULT, NULL);

    npy_intp* shape = (npy_intp*)malloc(2*sizeof(npy_intp));
    shape[0] = n;
    shape[1] = m;
    PyArray_Dims dims = {shape, 2};
    PyObject* gramNumpyReshape = PyArray_Newshape(gramNumpy, &dims, NPY_CORDER);
    free(shape);
    return gramNumpyReshape;
}

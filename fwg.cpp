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

double sliced_wasserstein_distance(
    embedding_t embedding_i,
    embedding_t embedding_j,
    int M
) {

    int size_i = embedding_i.size();
    int size_j = embedding_j.size();

    int k, l;
    int u = size_i + size_j;

    std::vector<double> vec1_1(u);
    std::vector<double> vec1_2(u);
    std::vector<double> vec2_1(u);
    std::vector<double> vec2_2(u);

    for (k=0; k<size_i; k++) {
        double birth = embedding_i[k].first;
        double death = embedding_i[k].second;
        vec1_1[k] = birth;
        vec1_2[k] = death;
        vec2_1[k] = (birth+death)/2.0;
        vec2_2[k] = (birth+death)/2.0;
    }

    for (k=0; k<size_j; k++) {
        double birth = embedding_j[k].first;
        double death = embedding_j[k].second;
        vec2_1[size_i+k] = birth;
        vec2_2[size_i+k] = death;
        vec1_1[size_i+k] = (birth+death)/2.0;
        vec1_2[size_i+k] = (birth+death)/2.0;
    }
    

    double sw = 0;
    double theta = - M_PI / 2.0;
    double s = M_PI / M;

    for (k=0; k<M; k++) {
    
        std::vector<double> v1(u);
        std::vector<double> v2(u);
        for (l=0; l<u; l++) {
            v1[l] = vec1_1[l] * cos(theta) + vec1_2[l] * sin(theta);
            v2[l] = vec2_1[l] * cos(theta) + vec2_2[l] * sin(theta);
        }
        std::sort(v1.begin(), v1.end());
        std::sort(v2.begin(), v2.end());

        double norm1 = 0.0;
        for (l=0; l<u; l++) {
            double raw_val = v1[l] - v2[l];
            if (isinf(raw_val)) {
                norm1 += DBL_MAX;
            }
            else if (!isnan(raw_val)) {
                norm1 += fabs(raw_val);
            }
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

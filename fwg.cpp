#include <Python.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <pthread.h>

#include <iostream>
#include <thread>
#include <vector>
#include <future>

#define min(a,b) (a<=b?a:b)
#define max(a,b) (a>=b?a:b)

pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

static int compare (void const *a, void const *b)
{
   double const *pa = (const double *) a;
   double const *pb = (const double *) b;

   if (isinf(*pa)) {
       return *pa > 0 ? 1 : -1;
   }
   if (isinf(*pb)) {
       return *pb > 0 ? -1 : 1;
   }

   return *pa < *pb ? -1 : 1;
}

double sliced_wasserstein_distance(
    PyListObject* embedding_i,
    PyListObject* embedding_j,
    int size_i,
    int size_j,
    int M
) {

    int k, l;
    int u = size_i + size_j;

    double* vec1_1 = (double *)malloc(u * sizeof(double));
    double* vec1_2 = (double *)malloc(u * sizeof(double));
    double* vec2_1 = (double *)malloc(u * sizeof(double));
    double* vec2_2 = (double *)malloc(u * sizeof(double));

    for (k=0; k<size_i; k++) {
        PyTupleObject* pt = (PyTupleObject *) PyList_GetItem((PyObject *)embedding_i, k);
        double birth = PyFloat_AS_DOUBLE(PyTuple_GetItem((PyObject *)pt, 0));
        double death = PyFloat_AS_DOUBLE(PyTuple_GetItem((PyObject *)pt, 1));
        vec1_1[k] = birth;
        vec1_2[k] = death;
        vec2_1[k] = (birth+death)/2.0;
        vec2_2[k] = (birth+death)/2.0;
    }

    for (k=0; k<size_j; k++) {
        PyTupleObject* pt = (PyTupleObject *) PyList_GetItem((PyObject *)embedding_j, k);
        double birth = PyFloat_AS_DOUBLE(PyTuple_GetItem((PyObject *)pt, 0));
        double death = PyFloat_AS_DOUBLE(PyTuple_GetItem((PyObject *)pt, 1));
        vec2_1[size_i+k] = birth;
        vec2_2[size_i+k] = death;
        vec1_1[size_i+k] = (birth+death)/2.0;
        vec1_2[size_i+k] = (birth+death)/2.0;
    }
    

    double sw = 0;
    double theta = - M_PI / 2.0;
    double s = M_PI / M;

    for (k=0; k<M; k++) {
    
        double* v1 = (double *)malloc(u * sizeof(double));
        double* v2 = (double *)malloc(u * sizeof(double));
        for (l=0; l<u; l++) {
            v1[l] = vec1_1[l] * cos(theta) + vec1_2[l] * sin(theta);
            v2[l] = vec2_1[l] * cos(theta) + vec2_2[l] * sin(theta);
        }
        
        qsort(v1, u, sizeof(double), compare);
        qsort(v2, u, sizeof(double), compare);

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

        free(v1);
        free(v2);

        sw = sw + s * norm1;
        theta = theta + s;
    }

    free(vec1_1);
    free(vec1_2);
    free(vec2_1);
    free(vec2_2);
    
    return (1 / M_PI) * sw;

}

PyListObject* fast_wasserstein_distances_single_thread(
    PyListObject* embeddings_in,
    PyListObject* embeddings_out,
    int M
)
 {

    int n = (int) PyList_Size((PyObject*)embeddings_in);
    int m = (int) PyList_Size((PyObject*)embeddings_out);
    int i, j;

    std::vector<std::future<double>> my_futures;

    PyListObject* gram = (PyListObject *) PyList_New(n*m);

    for (i=0; i<n; i++) {
        PyListObject* embedding_i = (PyListObject*) PyList_GetItem((PyObject *)embeddings_in, i);
        int size_i = (int) PyList_Size((PyObject*)embedding_i);

        my_futures.clear();

        for (j=0; j<m; j++) {
            PyListObject* embedding_j = (PyListObject*) PyList_GetItem((PyObject *)embeddings_out, j);
            int size_j = (int) PyList_Size((PyObject*)embedding_j);

            my_futures.push_back(std::async(std::launch::async, sliced_wasserstein_distance, embedding_i,
                embedding_j,
                size_i,
                size_j,
                M));
        }

        for (j=0; j<m; j++) {
            PyList_SET_ITEM(gram, i*n+j, PyFloat_FromDouble(my_futures[j].get()));
        }
    }

    return gram;

}

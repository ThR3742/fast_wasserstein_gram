#include <Python.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int compare (void const *a, void const *b)
{
   double const *pa = a;
   double const *pb = b;
   return *pa - *pb;
}


PyListObject* fast_wasserstein_gram(
    PyListObject* embeddings_in,
    PyListObject* embeddings_out,
    int M,
    double sigma
)
 {
    int n = (int) PyList_Size(embeddings_in);
    int m = (int) PyList_Size(embeddings_out);
    int i, j, k, l;

    PyListObject* gram = PyList_New(n);

    for (i=0; i<n; i++) {

        PyListObject* matrix_line = PyList_New(m);
        PyList_SET_ITEM(gram, i, matrix_line);

        for (j=0; j<m; j++) {

            PyListObject* embedding_i = PyList_GetItem(embeddings_in, i);
            PyListObject* embedding_j = PyList_GetItem(embeddings_out, j);

            int size_i = (int) PyList_Size(embedding_i);
            int size_j = (int) PyList_Size(embedding_j);

            int u = size_i + size_j;

            double* vec1_1 = (double *)malloc(u * sizeof(double));
            double* vec1_2 = (double *)malloc(u * sizeof(double));
            double* vec2_1 = (double *)malloc(u * sizeof(double));
            double* vec2_2 = (double *)malloc(u * sizeof(double));

            for (k=0; k<size_i; k++) {
                PyTupleObject* pt = PyList_GetItem(embedding_i, k);
                double birth = PyFloat_AS_DOUBLE(PyTuple_GetItem(pt, 0));
                double death = PyFloat_AS_DOUBLE(PyTuple_GetItem(pt, 1));
                vec1_1[k] = birth;
                vec1_2[k] = death;
                vec2_1[k] = (birth+death)/2.0;
                vec2_2[k] = (birth+death)/2.0;
            }

            for (k=0; k<size_j; k++) {
                PyTupleObject* pt = PyList_GetItem(embedding_j, k);
                double birth = PyFloat_AS_DOUBLE(PyTuple_GetItem(pt, 0));
                double death = PyFloat_AS_DOUBLE(PyTuple_GetItem(pt, 1));
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
                    v1[l] = (vec1_1[l] + vec1_2[l])*theta;
                    v2[l] = (vec2_1[l] + vec2_2[l])*theta;
                }
                qsort(v1, u, sizeof(double), compare);
                qsort(v2, u, sizeof(double), compare);

                double norm1 = 0.0;
                for (l=0; l<u; l++) {
                    norm1 += abs(v1[l] - v2[l]);
                }

                sw = sw + s * norm1;
                theta = theta + s;
            }
            
            double val = exp(-(1 / M_PI) * sw / pow(2 * sigma, 2));

            PyList_SET_ITEM(matrix_line, j, PyFloat_FromDouble(val));
        }
    }

    return gram;

}
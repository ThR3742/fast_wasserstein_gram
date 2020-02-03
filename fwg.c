#include <Python.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>

static int compare (void const *a, void const *b)
{
   double const *pa = a;
   double const *pb = b;

   if (isinf(*pa)) {
       return *pa > 0 ? 1 : -1;
   }
   if (isinf(*pb)) {
       return *pb > 0 ? -1 : 1;
   }

   return *pa < *pb ? -1 : 1;
}


PyListObject* fast_wasserstein_distance(
    PyListObject* embeddings_in,
    PyListObject* embeddings_out,
    int M
)
 {
    int n = (int) PyList_Size(embeddings_in);
    int m = (int) PyList_Size(embeddings_out);
    int i, j, k, l;

    PyListObject* gram = PyList_New(n);

    for (i=0; i<n; i++) {

        PyListObject* matrix_line = PyList_New(m);
        PyList_SET_ITEM(gram, i, matrix_line);

        PyListObject* embedding_i = PyList_GetItem(embeddings_in, i);
        int size_i = (int) PyList_Size(embedding_i);

        for (j=0; j<m; j++) {

            PyListObject* embedding_j = PyList_GetItem(embeddings_out, j);

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

                sw = sw + s * norm1;
                theta = theta + s;
            }
            
            double val = (1 / M_PI) * sw;

            PyList_SET_ITEM(matrix_line, j, PyFloat_FromDouble(val));
        }
    }

    return gram;

}
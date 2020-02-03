#include <Python.h>
#include <math.h>

PyListObject* fast_wasserstein_gram(
    PyListObject* embeddings_in,
    PyListObject* embeddings_out,
    int M,
    double sigma
)
 {
    int n = (int) PyList_Size(embeddings_in);
    int m = (int) PyList_Size(embeddings_out);
    int i;

    PyListObject* gram = PyList_New(n);

    int j, k;

    for (i=0; i<n; i++) {

        PyListObject* matrix_line = PyList_New(m);
        PyList_SET_ITEM(gram, i, matrix_line);

        for (j=0; j<m; j++) {

            PyListObject* embedding_i = PyList_GetItem(embeddings_in, i);
            PyListObject* embedding_j = PyList_GetItem(embeddings_out, j);

            int size_i = (int) PyList_Size(embedding_i);
            int size_j = (int) PyList_Size(embedding_j);

            int u = size_i + size_j;

            // printf("Computing product between %d and %d\n", size_i, size_j);

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

                
            }
           /**
            for k in range(M):
                v1 = [np.dot(pt1, (theta, theta)) for pt1 in vec1]
                v2 = [np.dot(pt2, (theta, theta)) for pt2 in vec2]
                v1.sort()
                v2.sort()
                val = np.asarray(v1) - np.asarray(v2)
                val[np.isnan(val)] = 0.0
                # val = np.nan_to_num(np.asarray(v1) - np.asarray(v2))
                sw = sw + s * np.linalg.norm(val, ord=1)
                theta = theta + s
                # logger.info(f"End Sliced Wass. Kernel")
                # print("Run :", i, " and sw =", (1/np.pi)*sw)
            gram[i, j] = np.exp(-(1 / np.pi) * sw / (2 * sigma ** 2))
        **/
            double val = 0.0;

            PyList_SET_ITEM(matrix_line, j, PyFloat_FromDouble(val));
        }
    }

    return gram;

}
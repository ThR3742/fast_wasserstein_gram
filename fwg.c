#include <Python.h>

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

    double **arr = (double **)malloc(n * sizeof(double *)); 
    for (i=0; i<n; i++) 
         arr[i] = (double *)malloc(m * sizeof(double));

    int j;

    for (i=0; i<n; i++) {

        PyListObject* matrix_line = PyList_New(m);
        PyList_SET_ITEM(gram, i, matrix_line);

        for (j=0; j<m; j++) {

            PyListObject* embedding_i = PyList_GetItem(embeddings_in, i);
            PyListObject* embedding_j = PyList_GetItem(embeddings_out, j);

            int size_i = (int) PyList_Size(embedding_i);
            int size_j = (int) PyList_Size(embedding_j);

            int u = size_i + size_j;

            printf("Computing product between %d and %d\n", size_i, size_j);

            
            /**
             * 
             * 
             * # logger.info(f"Sliced Wass. Kernel ")
            u = len(dgm1) + len(dgm2)
            vec1 = [(0.0, 0.0) for _ in range(u)]
            vec2 = [(0.0, 0.0) for _ in range(u)]
            for k, pt1 in enumerate(dgm1):
                vec1[k] = (pt1[0], pt1[1])
                vec2[k] = ((pt1[0] + pt1[1]) / 2.0, (pt1[0] + pt1[1]) / 2.0)
            for k, pt2 in enumerate(dgm2):
                vec2[k + len(dgm1)] = (pt2[0], pt2[1])
                vec1[k + len(dgm1)] = ((pt2[0] + pt2[1]) / 2.0, (pt2[0] + pt2[1]) / 2.0)
            sw = 0
            theta = -np.pi / 2
            s = np.pi / M
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
#ifndef FWG_H
#define FWG_H

PyObject* fast_wasserstein_distances(
    PyListObject* embeddings_in,
    PyListObject* embeddings_out,
    int M
);


#endif //FWG_H

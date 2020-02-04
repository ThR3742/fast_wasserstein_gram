#ifndef FWG_H
#define FWG_H

PyListObject* fast_wasserstein_distances(
    PyListObject* embeddings_in,
    PyListObject* embeddings_out,
    int M
);

#endif //FWG_H

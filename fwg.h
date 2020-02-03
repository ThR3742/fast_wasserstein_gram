#ifndef FWG_H
#define FWG_H

PyListObject* fast_wasserstein_gram(
    PyListObject* embeddings_in,
    PyListObject* embeddings_out,
    int M,
    double sigma
);

#endif //FWG_H

#ifndef FWG_H
#define FWG_H

#include <vector>
#include <utility>

typedef std::vector<std::pair<double, double>> embedding_t;

PyObject* fast_wasserstein_distances(
    std::vector<embedding_t> embeddings_in,
    std::vector<embedding_t> embeddings_out,
    int M
);


#endif //FWG_H

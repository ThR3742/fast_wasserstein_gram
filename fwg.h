#ifndef FWG_H
#define FWG_H

PyListObject* fast_wasserstein_distances(
    PyListObject* embeddings_in,
    PyListObject* embeddings_out,
    int M,
    int max_jobs
);

PyListObject* fast_wasserstein_distances_single_thread(
    PyListObject* embeddings_in,
    PyListObject* embeddings_out,
    int M
);


#endif //FWG_H

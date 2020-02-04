#include <Python.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <pthread.h>

#define min(a,b) (a<=b?a:b)
#define max(a,b) (a>=b?a:b)

pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

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

    
    pthread_mutex_lock(&lock);
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
    pthread_mutex_unlock(&lock);

    

    
    
    return (1 / M_PI) * sw;

}

struct chunk {
    int start_line_index;
    int end_line_index;
};

struct thread_args {
    int chunk_number;
    int start_line_index;
    int end_line_index;
    PyListObject* embeddings_in;
    PyListObject* embeddings_out;
    int M;
    PyListObject** matrix_lines;
};

struct chunk* list_chunks(int size_t, int nb_chunks, int size_chunk) {
    //printf("Using %d chunks of size %d for total size %d\n", nb_chunks, size_chunk, size_t);
    struct chunk *ret = (struct chunk*) malloc(nb_chunks*sizeof(struct chunk));
    int c;
    for (c=0; c<nb_chunks; c++) {
        struct chunk my_chunk = {c*size_chunk, min((c+1)*size_chunk, size_t)};
        //printf("Chunk [%d, %d]\n", my_chunk.start_line_index, my_chunk.end_line_index);
        ret[c] = my_chunk;
    }
    return ret;
}

void compute_chunk(void *args) {

    struct thread_args *my_args = (struct thread_args*)args;

    printf("Computing chunk %d from %d to %d\n",
            my_args->chunk_number,
            my_args->start_line_index,
            my_args->end_line_index
            );


    if (my_args->end_line_index > my_args->start_line_index) {
        int m = (int) PyList_Size(my_args->embeddings_out);
        int i, j;
        int chunk_size = my_args->end_line_index - my_args->start_line_index;
        for (i=0; i < my_args->end_line_index - my_args->start_line_index; i++) {
            //printf("[%d] i=%d\n", my_args->chunk_number, i);
            PyListObject* matrix_line = my_args->matrix_lines[i+my_args->start_line_index];
            PyListObject* embedding_i = PyList_GET_ITEM(my_args->embeddings_in, i+my_args->start_line_index);
            int size_i = (int) PyList_Size(embedding_i);
            for (j=0; j<m; j++) {
                PyListObject* embedding_j = PyList_GET_ITEM(my_args->embeddings_out, j);
                int size_j = (int) PyList_Size(embedding_j);
                
                double val = sliced_wasserstein_distance(
                    embedding_i,
                    embedding_j,
                    size_i,
                    size_j,
                    my_args->M
                );
                PyList_SetItem(matrix_line, j, PyFloat_FromDouble(val));
            }
        }
    }




    printf("Chunk %d from %d to %d done !\n",
            my_args->chunk_number,
            my_args->start_line_index,
            my_args->end_line_index
            );

    pthread_exit(NULL);

}


PyListObject* fast_wasserstein_distances(
    PyListObject* embeddings_in,
    PyListObject* embeddings_out,
    int M,
    int max_jobs
)
 {

    int n = (int) PyList_Size(embeddings_in);
    int m = (int) PyList_Size(embeddings_out);
    PyListObject* gram = PyList_New(n);
    int nb_chunks = min(max_jobs, n);

    printf("Using %d chunks...\n", nb_chunks);
    pthread_t* threads = (pthread_t*)malloc(nb_chunks*sizeof(pthread_t));

    int size_chunk = max(ceil((double) n / nb_chunks), 1);
    struct chunk* my_chunks = list_chunks(n, nb_chunks, size_chunk);

    int i, c;

    struct thread_args *all_thread_args = (struct thread_args*)malloc(nb_chunks*sizeof(struct thread_args));

    PyListObject** matrix_lines = (PyListObject**) malloc(n*sizeof(PyListObject*));

    for (i=0; i<n ; i++) {
        PyListObject* matrix_line = PyList_New(m);
        PyList_SetItem(gram, i, matrix_line);
        matrix_lines[i] = matrix_line;
    }


    for (c=0; c<nb_chunks; c++) {
        struct thread_args my_arg = {
                c,
                my_chunks[c].start_line_index,
                my_chunks[c].end_line_index,
                embeddings_in,
                embeddings_out,
                M,
                matrix_lines
        };
        all_thread_args[c] = my_arg;
    }

    for (c=0; c<nb_chunks; c++) {
        pthread_t my_thread;
        //printf("Creating thread %d (%d)\n", c, my_thread);
        //printf("It should handle %d to %d...\n", my_chunks[c].start_line_index, my_chunks[c].end_line_index);
        int rc = pthread_create(&my_thread, NULL, &compute_chunk, (void *) &(all_thread_args[c]));
        //printf("Creating thread %d with return status %d\n", c, rc);
        threads[c] = my_thread;
    }

    for (c=0; c<nb_chunks; c++) {
        int rc = pthread_join(threads[c], NULL);
        if (rc) {
            printf("[ERROR Thread %d (%d)] From pthread_join: %d\n", c, threads[c], rc);
        }
        else {
            //printf("Thread %d (%d) joined !\n", c, threads[c]);
        }
    }

    //printf("Job done !!\n");
    
    return gram;
}


PyListObject* fast_wasserstein_distances_single_thread(
    PyListObject* embeddings_in,
    PyListObject* embeddings_out,
    int M
)
 {
    int n = (int) PyList_Size(embeddings_in);
    int m = (int) PyList_Size(embeddings_out);
    int i, j;

    PyListObject* gram = PyList_New(n);

    for (i=0; i<n; i++) {

        PyListObject* matrix_line = PyList_New(m);
        PyList_SET_ITEM(gram, i, matrix_line);

        PyListObject* embedding_i = PyList_GetItem(embeddings_in, i);
        int size_i = (int) PyList_Size(embedding_i);

        for (j=0; j<m; j++) {

            PyListObject* embedding_j = PyList_GetItem(embeddings_out, j);

            int size_j = (int) PyList_Size(embedding_j);

            double val = sliced_wasserstein_distance(
                embedding_i,
                embedding_j,
                size_i,
                size_j,
                M
            );

            PyList_SET_ITEM(matrix_line, j, PyFloat_FromDouble(val));
        }
    }

    return gram;

}

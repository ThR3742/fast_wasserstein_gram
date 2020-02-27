#include <Python.h>
#include "fwg.h"
#include <vector>
#include <utility>

static std::vector<embedding_t> convertList(
    PyObject* pyo_list
) {
    int nb_embeddings = (int) PyList_Size((PyObject*)pyo_list);
    std::vector<std::vector<std::pair<double, double>>> cpp_list(nb_embeddings);

    for (int i=0; i<nb_embeddings; ++i) {
        PyObject* embedding = PyList_GetItem(pyo_list, i);
        int nb_points = (int) PyList_Size(embedding);

        embedding_t emb(nb_points);        

        for (int j=0; j<nb_points; ++j) {
            PyObject* pt = PyList_GetItem(embedding, j);
            double birth = PyFloat_AS_DOUBLE(PyTuple_GetItem(pt, 0));
            double death = PyFloat_AS_DOUBLE(PyTuple_GetItem(pt, 1));
            emb[j] = {birth, death};
        }

        cpp_list[i] = emb;
    }
    return cpp_list;
}

static PyObject *
fwd_call(PyObject *self, PyObject *args)
{
    // Fetching arguments
    PyObject* pyo_embeddings_in = PyTuple_GetItem(args, 0);
    PyObject* pyo_embeddings_out = PyTuple_GetItem(args, 1);
    PyObject* pyo_m = PyTuple_GetItem(args, 2);

    // Casting everybody to native c++ types
    long m = PyLong_AsLong(pyo_m);

    std::vector<embedding_t> embeddings_in = convertList(pyo_embeddings_in);
    std::vector<embedding_t> embeddings_out = convertList(pyo_embeddings_out);
    
    PyObject* gram = fast_wasserstein_distances(
        embeddings_in,
        embeddings_out,
        m
    );

    return gram;
}


static PyMethodDef FwgMethods[] = {
    {"fwd",  fwd_call, METH_VARARGS,
     "Compute Wasserstein Distance Matrix in an optimized way"},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef fwgmodule = {
    PyModuleDef_HEAD_INIT,
    "fwg",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    FwgMethods
};

PyMODINIT_FUNC
PyInit_fwg(void)
{
    return PyModule_Create(&fwgmodule);
}
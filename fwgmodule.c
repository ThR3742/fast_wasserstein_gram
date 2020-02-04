#include <Python.h>
#include "fwg.h"

static PyObject *
fwd_call(PyObject *self, PyObject *args)
{

    static char *kwlist[] = {"embeddings_in", "embeddings_out", "M", NULL};


    PyListObject* pyo_embeddings_in = PyTuple_GetItem(args, 0);
    PyListObject* pyo_embeddings_out = PyTuple_GetItem(args, 1);
    PyObject* pyo_m = PyTuple_GetItem(args, 2);
    //PyObject* pyo_max_jobs = PyTuple_GetItem(args, 3);

    long m = PyLong_AsLong(pyo_m);
    //long max_jobs = PyLong_AsLong(pyo_max_jobs);

    int n_size = (int) PyList_Size(pyo_embeddings_in);
    int m_size = (int) PyList_Size(pyo_embeddings_out);

    PyListObject* gram = fast_wasserstein_distances_single_thread(
        pyo_embeddings_in,
        pyo_embeddings_out,
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
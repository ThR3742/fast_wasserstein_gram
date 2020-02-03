#include <Python.h>
#include "fwg.h"

static PyObject *
fwg_call(PyObject *self, PyObject *args)
{

    static char *kwlist[] = {"embeddings_in", "embeddings_out", "M", NULL};

    int M;

    PyListObject* pyo_embeddings_in = PyTuple_GetItem(args, 0);
    PyListObject* pyo_embeddings_out = PyTuple_GetItem(args, 1);
    PyObject* pyo_m = PyTuple_GetItem(args, 2);
    PyObject* pyo_sigma = PyTuple_GetItem(args, 3);

    long m = PyLong_AsLong(pyo_m);
    double sigma = PyFloat_AsDouble(pyo_sigma);

    int n_size = (int) PyList_Size(pyo_embeddings_in);
    int m_size = (int) PyList_Size(pyo_embeddings_out);

    printf("[FWG] m=%i\n", m);
    printf("[FWG] sigma=%f\n", sigma);
    printf("[FWG] embeddings_in has %d items\n", n_size);
    printf("[FWG] embeddings_out has %d items\n", m_size);

    PyListObject* gram = fast_wasserstein_gram(
        pyo_embeddings_in,
        pyo_embeddings_out,
        M,
        sigma
    );

    Py_INCREF(gram);

    return gram;
}

static PyMethodDef FwgMethods[] = {
    {"fwg",  fwg_call, METH_VARARGS,
     "Compute Wasserstein Gram Matrix in an optimized way"},
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
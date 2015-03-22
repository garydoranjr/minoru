/*****************************/
/* Python Wrapper for libcam */
/*****************************/
#include <Python.h>
#include <numpy/arrayobject.h>

#include <libcam.h>

static PyObject *capture(PyObject *self, PyObject *args) {
    PyArrayObject *retleft, *retright;

    char *file1=NULL, *file2=NULL;
    int w, h, fps;
    double *data;

    // Parse inputs
    if (!PyArg_ParseTuple(args, "ssiii",
        &file1, &file2, &w, &h, &fps)) {
        return NULL;
    }
    if (file1 == NULL || file2 == NULL) {
        return NULL;
    }

    // Set up array for result
    npy_intp dims[3];
    dims[0] = (npy_intp) h;
    dims[1] = (npy_intp) w;
    dims[2] = (npy_intp) 3;
    retleft  = (PyArrayObject *)PyArray_SimpleNew(3, dims, NPY_DOUBLE);
    retright = (PyArrayObject *)PyArray_SimpleNew(3, dims, NPY_DOUBLE);

    // Set up memory for result
    data = (double *) retleft->data;

    return Py_BuildValue("NN", (PyObject *)retleft, (PyObject *)retright);
}

static PyMethodDef c_minoru_methods[] = {
   { "c_capture", (PyCFunction)capture, METH_VARARGS, "Capture (Left, Right) Video Frame"},
   { NULL, NULL, 0, NULL }
};

extern "C" void initc_minoru(void)
{
    Py_InitModule3("c_minoru", c_minoru_methods,
                   "Python wrapper for libcam");
    import_array();
}

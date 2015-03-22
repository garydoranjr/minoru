/*****************************/
/* Python Wrapper for libcam */
/*****************************/
#include <Python.h>
#include <numpy/arrayobject.h>

#include <libcam.h>

static PyObject *capture(PyObject *self, PyObject *args) {
    PyArrayObject *retleft, *retright;

    char *lFile=NULL, *rFile=NULL;
    int w, h, fps, init;
    double *data;

    // Parse inputs
    if (!PyArg_ParseTuple(args, "ssiiii",
        &lFile, &rFile, &w, &h, &fps, &init)) {
        return NULL;
    }
    if (lFile == NULL || rFile == NULL) {
        return NULL;
    }

    // Set up array for result
    npy_intp dims[3];
    dims[0] = (npy_intp) h;
    dims[1] = (npy_intp) w;
    dims[2] = (npy_intp) 3;
    retleft  = (PyArrayObject *)PyArray_SimpleNew(3, dims, NPY_UBYTE);
    retright = (PyArrayObject *)PyArray_SimpleNew(3, dims, NPY_UBYTE);

    // Open Cameras
    Camera cl(lFile, w, h, fps); //left camera object using libv4lcam2
    Camera cr(rFile, w, h, fps); //right camera object
    cl.Update(&cr);
    cl.toArray((unsigned char *) retleft->data);
    cr.toArray((unsigned char *) retright->data);

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

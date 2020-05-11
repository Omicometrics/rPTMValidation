#include <Python.h>

#include "annotate.h"
#include "crPTMDetermine.h"

// Boilerplate code for C++ extension

static PyMethodDef crPTMDetermine_methods[] = {
	{"annotate", python_annotate, METH_VARARGS, "Spectrum annotation."},
	{NULL, NULL, 0, NULL} /* SENTINEL */
};

PyDoc_STRVAR(crptmdetermineModuleDoc, "CPython functions for rPTMDetermine");

static struct PyModuleDef crPTMDetermineExt = {
	PyModuleDef_HEAD_INIT,
	"crPTMDetermine",
	crptmdetermineModuleDoc,
	-1,
	crPTMDetermine_methods
};

PyMODINIT_FUNC PyInit_crPTMDetermine(void) {
	return PyModule_Create(&crPTMDetermineExt);
}
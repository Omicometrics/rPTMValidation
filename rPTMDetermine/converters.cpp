#include <Python.h>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "annotation.h"
#include "converters.h"

/* Python to C */

std::vector<Ion> tupleListToIonVector(PyObject* source) {
	std::vector<Ion> data;
	if (PyList_Check(source)) {
		for (Py_ssize_t ii = 0; ii < PyList_Size(source); ii++) {
			PyObject* value = PyList_GetItem(source, ii);
			if (PyTuple_Check(value)) {
				data.emplace_back(value);
			}
			else {
				throw std::logic_error("Contained PyObject pointer was not a tuple");
			}
		}
	} 
	else {
		throw std::logic_error("PyObject pointer was not a list");
	}
	
	return data;
}

std::vector<double> listToDoubleVector(PyObject* source) {
	std::vector<double> data;
	if (PyList_Check(source)) {
		for (Py_ssize_t ii = 0; ii < PyList_Size(source); ii++) {
			PyObject* value = PyList_GetItem(source, ii);
			if (PyFloat_Check(value)) {
				data.push_back(PyFloat_AsDouble(value));
			}
			else {
				throw std::logic_error("Contained PyObject pointer was not a float");
			}
		}
	}
	else {
		throw std::logic_error("PyObject pointer was not a list");
	}
	
	return data;
}

/* C to Python */

PyObject* annotationMapToPyDict(const std::map<std::string, Annotation>& anns) {
	PyObject* dictObj = PyDict_New();
	
	for (const std::pair<std::string, Annotation>& ann : anns) {
		PyDict_SetItemString(
			dictObj,
			ann.first.c_str(),
			(PyObject*)(Annotation) ann.second);
	}
	
	return dictObj;
}
#include <Python.h>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "annotation.h"
#include "converters.h"

/* Python to C++ */

bool checkFloat(PyObject* obj) {
	return PyFloat_Check(obj);
}

bool checkString(PyObject* obj) {
	return PyUnicode_Check(obj);
}

bool checkBool(PyObject* obj) {
	return PyBool_Check(obj);
}

std::string unicodeToString(PyObject* obj) {
	return std::string(PyUnicode_AsUTF8(obj));
}

template<class T>
std::vector<T> listToVector(PyObject* source, bool(*check)(PyObject*), T(*convert)(PyObject*)) {
	if (!PyList_Check(source)) {
		throw std::logic_error("PyObject pointer was not a list");
	}
	
	long size = (long) PyList_Size(source);
	std::vector<T> data;
	data.reserve(size);
	for (Py_ssize_t ii = 0; ii < size; ii++) {
		PyObject* value = PyList_GetItem(source, ii);
		if (check(value)) {
			data.push_back(convert(value));
		}
		else {
			throw std::logic_error("Contained PyObject pointer was not expected type");
		}
	}
	return data;
}

std::vector<double> listToDoubleVector(PyObject* source) {
	return listToVector<double>(source, &checkFloat, &PyFloat_AsDouble);
}

std::vector<std::string> listToStringVector(PyObject* source) {
	return listToVector<std::string>(source, &checkString, &unicodeToString);
}

std::vector<int> listToBoolVector(PyObject* source) {
	return listToVector<int>(source, &checkBool, &PyObject_IsTrue);
}

std::vector<Ion> tupleListToIonVector(PyObject* source) {
	std::vector<Ion> data;
	if (PyList_Check(source)) {
		long size = PyList_Size(source);
		data.reserve(size);
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

/* C to Python */

PyObject* longVectorToList(const std::vector<long>& data) {
	PyObject* listObj = PyList_New(data.size());

	int size = (int) data.size();

	for (int ii = 0; ii < size; ii++) {
		PyList_SetItem(listObj, ii, PyLong_FromLong(data[ii]));
	}
	
	return listObj;
}

PyObject* annotationMapToPyDict(const std::map<std::string, Annotation>& anns) {
	PyObject* dictObj = PyDict_New();
	
	for (const std::pair<std::string, Annotation>& ann : anns) {
	    PyObject* annotation = (PyObject*)(Annotation) ann.second;
		PyDict_SetItemString(
			dictObj,
			ann.first.c_str(),
			annotation
		);
		Py_DECREF(annotation);
	}
	
	return dictObj;
}
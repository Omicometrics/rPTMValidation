#ifndef _RPTMDETERMINE_ANNOTATION_H
#define _RPTMDETERMINE_ANNOTATION_H

#include <Python.h>
#include <string>

struct Annotation {
	long index;
	double delta_mass;
	long position;
	
	Annotation(long _index, double _delta_mass, long _position)
		: index(_index), delta_mass(_delta_mass), position(_position) {}
		
	explicit operator PyObject*() const {
	    PyObject* tuple = PyTuple_New(3);
	    PyTuple_SetItem(tuple, 0, PyLong_FromLong(index));
	    PyTuple_SetItem(tuple, 1, PyFloat_FromDouble(delta_mass));
	    PyTuple_SetItem(tuple, 2, PyLong_FromLong(position));
	    return tuple;
	}
};

struct Ion {
	double mass;
	std::string label;
	long position;
	
	Ion(double _mass, const std::string& _label, long _position)
		: mass(_mass), label(_label), position(_position) {}
		
	Ion(PyObject* pySource) 
		: mass(PyFloat_AsDouble( PyTuple_GetItem(pySource, 0) )),
		  label(PyUnicode_AsUTF8( PyTuple_GetItem(pySource, 1) )),
		  position(PyLong_AsLong( PyTuple_GetItem(pySource, 2) )) {}
};

#endif // _RPTMDETERMINE_ANNOTATION_H
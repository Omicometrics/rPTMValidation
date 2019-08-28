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
		
	explicit operator PyObject*() {
		return PyTuple_Pack(
			3,
			PyLong_FromLong(index),
			PyFloat_FromDouble(delta_mass),
			PyLong_FromLong(position)
		);
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
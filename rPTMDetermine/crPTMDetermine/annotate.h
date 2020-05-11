#ifndef _RPTMDETERMINE_ANNOTATE_H
#define _RPTMDETERMINE_ANNOTATE_H

#include <Python.h>
#include <map>
#include <string>
#include <vector>

#include "annotation.h"

std::map<std::string, Annotation> annotate(
	const std::vector<double>& mzArray,
	const std::vector<Ion>& theorIons,
	const double tol);
	
extern "C" {
	PyObject* python_annotate(PyObject* module, PyObject* args);
}

#endif // _RPTMDETERMINE_ANNOTATE_H
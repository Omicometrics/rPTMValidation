#ifndef _RPTMDETERMINE_CONVERTERS_H
#define _RPTMDETERMINE_CONVERTERS_H

#include <Python.h>
#include <map>
#include <string>
#include <vector>

#include "annotation.h"

std::vector<Ion> tupleListToIonVector(PyObject* source);

std::vector<double> listToDoubleVector(PyObject* source);

PyObject* annotationMapToPyDict(const std::map<std::string, Annotation>& anns);

#endif // _RPTMDETERMINE_CONVERTERS_H
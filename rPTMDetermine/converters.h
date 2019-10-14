#ifndef _RPTMDETERMINE_CONVERTERS_H
#define _RPTMDETERMINE_CONVERTERS_H

#include <Python.h>
#include <map>
#include <string>
#include <vector>

#include "annotation.h"

std::vector<Ion> tupleListToIonVector(PyObject* source);

std::vector<double> listToDoubleVector(PyObject* source);

std::vector<std::string> listToStringVector(PyObject* source);

std::vector<int> listToBoolVector(PyObject* source);

PyObject* longVectorToList(const std::vector<long>& data);

PyObject* annotationMapToPyDict(const std::map<std::string, Annotation>& anns);

#endif // _RPTMDETERMINE_CONVERTERS_H
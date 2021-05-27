#include <Python.h>
#include <cmath>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "annotate.h"
#include "annotation.h"
#include "converters.h"
	
std::map<std::string, Annotation> annotate(
		const std::vector<double>& mzArray,
		const std::vector<Ion>& theorIons,
		const double tol)
{
	std::map<std::string, Annotation> anns = std::map<std::string, Annotation>();
	
	int startIdx = 0;
	for (const Ion& ion : theorIons) {
		for (unsigned long idx = 0; idx < mzArray.size(); idx++) {
			double mz = mzArray[idx];
			double delta = ion.mass - mz;
			if (abs(delta) <= tol) {
				anns.emplace(std::piecewise_construct,
							 std::forward_as_tuple(ion.label),
							 std::forward_as_tuple(idx, delta, ion.position));
				startIdx = idx;
				break;
			}
			if (mz > ion.mass + tol) {
				startIdx = idx;
				break;
			}
		}
	}

	return anns;
}

PyObject* python_annotate(PyObject* module, PyObject* args) {
	PyObject* mzList = NULL;
	PyObject* theorIons = NULL;
	PyObject* tol = NULL;
	
	if (!PyArg_UnpackTuple(args, "cpython_annotate", 3, 3, &mzList, &theorIons, &tol)) return NULL;

	try {
	    return annotationMapToPyDict(
            annotate(
                listToDoubleVector(mzList),
                tupleListToIonVector(theorIons),
                PyFloat_AsDouble(tol)
            )
        );
	}
	catch (const std::exception& ex) {
	    PyErr_SetString(PyExc_RuntimeError, ex.what());
        return NULL;
	}
}

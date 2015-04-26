/*
 * Normalize.h
 *
 *  Created on: 15 avr. 2015
 *      Author: jfellus
 */

#ifndef NORMALIZE_H_
#define NORMALIZE_H_

#include <pg.h>
#include <matrix.h>

class Normalize {
public:
	Matrix* out;
	OUTPUT(Matrix, *out);

public:
	Normalize() {out = 0;}

	void init() {}

	void process(Matrix& m) {
		for(uint i=m.h;i--;) normalize(m.get_row(i), m.w);
		out = &m;
	}

private:

	float n2(float *v, uint dim) {
		float s = 0;
		for(uint i=dim; i--;) s+=v[i]*v[i];
		return sqrt(s);
	}

	void normalize(float* v, uint dim) {
		float l2 = n2(v,dim);
		for(uint i=dim; i--;) v[i]/=l2;
	}
};




#endif /* NORMALIZE_H_ */

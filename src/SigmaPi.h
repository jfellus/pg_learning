/*
 * SigmaPi.h
 *
 *  Created on: 15 avr. 2015
 *      Author: jfellus
 */

#ifndef SIGMAPI_H_
#define SIGMAPI_H_


#include <pg.h>
#include <matrix.h>

class SigmaPi {
public:
	Matrix out;
	OUTPUT(Matrix, out)

public:
	SigmaPi() {}

	void init() {}

	// REMARK : a and b must have the same height (number of observations)
	void process(const Matrix& a, const Matrix& b) {
		if(!out) { out.init(a.w,b.w); }
		float max = 0;
		//#pragma omp parallel for
		for(uint i=0; i<a.w; i++) {
			for(uint j=b.w; j--;) {
				out(i,j) = 0;
				for(uint k=0; k<a.h; k++) {
					out(i,j) += a(k,i)*b(k,j);
				}
				out(i,j) /= a.h;
				max = MAX(max, out(i,j));
 			}
		}

		for(uint i=0; i<out.n; i++) out[i]/=max;
	}
};



#endif /* SIGMAPI_H_ */

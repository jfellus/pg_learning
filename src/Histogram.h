/*
 * Histogram.h
 *
 *  Created on: 15 avr. 2015
 *      Author: jfellus
 */

#ifndef HISTOGRAM_H_
#define HISTOGRAM_H_

#include <pg.h>
#include <matrix.h>

class Histogram {
public:
	uint bins;
	bool use_weights;

	Matrix out;
	OUTPUT(Matrix, out)
public:
	Histogram() {
		bins = 1000;
		use_weights = true;
	}

	void init() {out.init(1, bins);}

	void process(const Matrix& in) {
		out = 0;
		if(in.w==1 || !use_weights) {
			for(uint i=in.n; i--;) {
				int v = (int)in[i];
				if(v>=0 && v<(int)bins) out[v]++;
			}
		} else {
			for(uint i=in.h; i--;) {
				int v = (int)in[i*in.w];
				float weight = in[i*in.w+1];
				if(v>=0 && v<(int)bins) out[v]+=weight;
			}
		}
	}
};


#endif /* HISTOGRAM_H_ */

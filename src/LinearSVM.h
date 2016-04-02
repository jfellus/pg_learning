/*
 * LinearSVM.h
 *
 *  Created on: 3 juin 2015
 *      Author: jfellus
 */

#ifndef LINEARSVM_H_
#define LINEARSVM_H_

#include <pg.h>

class LinearSVM {
public:

	float lambda;
	float eta;
	bool bRandom;
	bool bLearn;

	int label_positive;
	int label_negative;

	float nplus, nmoins;

	Matrix out;
	OUTPUT(Matrix,out);

	Matrix w;

	PARAM(float, lambda, {});
	PARAM(float, eta, {});
	PARAM(bool, bRandom, {});
	PARAM(bool, bLearn, {});
	PARAM(float, nplus, {});
	PARAM(float, nmoins, {});

public:

	LinearSVM() {
		nplus = nmoins = 0;
		lambda = 0.00001;
		eta = 0.0001;
		bRandom = true;
		label_positive = 1;
		label_negative = 0;
		bLearn = true;
	}

	void init() {}

	void process(const Matrix& X, const Matrix& labels) {
		if(!out) {
			out.init(X.h,1);
			w.init(1,X.w+1);
			w.randf(-1,1);
		}

		nplus = 0; nmoins = 0;
		for(uint i=0; i<X.h; i++) {
			const float* sample = X.get_row(i);

			float d = 0;
			for(uint x=0; x<X.w; x++) d += sample[x]*w[x];
			d += w[X.w];

			// Inference
			out[i]=(d>0);
			if(d>0) nplus++; else nmoins++;

			// SGD
			if(bLearn) {
				float label = labels[i]==label_positive ? 1 : -1;
				out[i]=(labels[i]==label_positive);
				if(label*d > 1) {
					for(uint x=0; x<X.w+1; x++) w[x] -= eta*lambda*w[x];
				} else {
					DBG("LOSS");
					for(uint x=0; x<X.w; x++) w[x] -= eta*(lambda*w[x] - label*sample[x]);
					w[X.w] -= eta*(lambda*w[X.w] - label);
				}
			}
		}

		DBG("+" << nplus << "\t\t-" << nmoins);
	}
};



#endif /* LINEARSVM_H_ */

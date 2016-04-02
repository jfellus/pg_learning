/*
 * NNClassifier.h
 *
 *  Created on: 28 avr. 2015
 *      Author: jfellus
 */

#ifndef NNCLASSIFIER_H_
#define NNCLASSIFIER_H_


#include <pg.h>
#include <matrix.h>
#include <float.h>
#include <vector>


class NNClassifier {
public:

	std::string prototypes_file;

	Matrix winners;
	Matrix prototypes;

	Matrix histogram;

	OUTPUT(Matrix, winners);

public:
	NNClassifier() {
		prototypes_file = "";
	}

	void init() {
		if(!prototypes_file.empty()) read_matrix(prototypes_file, prototypes);
		histogram.init(prototypes.h, 1);
	}

	void process(const Matrix& inputs) {
		if(winners.h != inputs.h) { winners.free(); winners.init(inputs.h, 1); }

		histogram.clear();

#pragma omp parallel for
		for(uint i=0; i<inputs.h; i++) {
			winners[i] = 0;
			float mind = dist(&prototypes(0,0), &inputs(i,0));
			for(uint k=1; k<prototypes.h; k++) {
				float d = dist(&prototypes(k,0), &inputs(i,0));
				if(d < mind) { winners[i] = k; mind = d; }
			}

//			histogram[winners[i]]++;
		}

//		histogram.dump();
	}


private:

	// Vector operations

	float dist(const float* a, const float* b) {
		float d = 0;
		for(uint i=prototypes.w; i--;) d += (a[i]-b[i])*(a[i]-b[i]);
		return d;
	}
};


#endif /* NNCLASSIFIER_H_ */

/*
 * SVQ_R.h
 *
 *  Created on: 15 avr. 2015
 *      Author: jfellus
 */

#ifndef SVQ_R_H_
#define SVQ_R_H_

#include <pg.h>
#include <matrix.h>
#include <float.h>


/**
 * NOTE : Uses L_2 distance
 * NOTE : Output activities are piecewise linear (not gaussian !)
 * IMPORTANT : Input pattern are always assumed in [0,1]^D  (otherwise, output activities of SVQ_R will be negative !!)
 */
class SVQ_R {
public:

	uint nbMaxClusters;
	float vigilance;
	Matrix prototypes;

	Matrix activities;
	Matrix winners;
	OUTPUT(Matrix, activities)
	OUTPUT(uint, activities.h)

public:

	SVQ_R() {
		nbMaxClusters = 1000;
		vigilance = 0.2;
	}

	void init() {}

	void process(const Matrix& inputs, float vigilance) {
		if(!activities) {
			prototypes.init(nbMaxClusters, inputs.w); prototypes.h=0;
			activities.init(nbMaxClusters, 1); activities.h=0;
			winners.init(inputs.h, 1);
		}

		// Compute winners and recruit
		winners.h = inputs.h;
		#pragma omp parallel for
		for(uint i=0; i<inputs.h; i++) {
			const float* x = inputs.get_row(i);
			int winner = compute_winner(x);
			if(winner == -1) {winner = recruit(x); DBG("RECRUIT " << prototypes.h);}
			winners[i] = winner;
		}

		// Compute output activities
		if(inputs.h) {
			for(uint j=activities.w;j--;) activities[j] = compute_activity(dist(prototypes.get_row(j),inputs.get_row(0),prototypes.w));
		}

	}

	void process(const Matrix& inputs) { process(inputs, vigilance); }

private:
	float dist(const float* a, const float* b, uint dim) {
		float d = 0;
		for(uint i=dim; i--;) d += fabs(a[i]-b[i]);
		return d / dim;
	}

	int compute_winner(const float* x) {
		int winner = -1;
		float min_dist = FLT_MAX;
		for(uint k=0; k<prototypes.h; k++) {
			float d = dist(prototypes.get_row(k), x, prototypes.w);
			if(d <= min_dist) { min_dist = d; winner = k; }
		}
		//DBG(min_dist);
		if(compute_vigilance(compute_activity(min_dist))) return -1;
		return winner;
	}

	inline float compute_activity(float dist) { return 1 - dist; }

	inline bool compute_vigilance(float activity) {	return activity < vigilance; }

	int recruit(const float* x) {
		if(prototypes.h<nbMaxClusters) {
			memcpy(prototypes.get_row(prototypes.h), x, prototypes.w);
			activities.h++;
			return prototypes.h++;
		}
		return prototypes.h-1;
	}
};



#endif /* SVQ_R_H_ */

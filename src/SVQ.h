/*
 * SVQ.h
 *
 *  Created on: 15 avr. 2015
 *      Author: jfellus
 */

#ifndef SVQ_H_
#define SVQ_H_

#include <pg.h>
#include <matrix.h>
#include <float.h>


/**
 * NOTE : Uses L_2 distance
 * NOTE : Output activities are piecewise linear (not gaussian !)
 * IMPORTANT : Input pattern are always assumed in [0,1]^D  (otherwise, output activities of SVQ_R will be negative !!)
 */
class SVQ {
public:

	uint K;
	float eta;

	Matrix prototypes;

	Matrix activities;
	Matrix winners;
	OUTPUT(Matrix, activities)
	OUTPUT(uint, activities.h)

public:

	SVQ() {
		eta = 0.01;
		K = 64;
	}

	void init() {}

	void process(const Matrix& inputs) {
		if(inputs.h > winners.h) {	activities.free();	winners.free();	}
		if(!prototypes) {prototypes.init(K, inputs.w); prototypes.set_height(0);}
		if(!activities) {
			activities.init(inputs.h, K);
			winners.init(inputs.h, 2);
			activities.set_height(0);
			winners.set_height(0);
		}

		// Compute and Adapt winners
		// NOTE : Prototypes initialization of  by recruiting the K first inputs
		winners.set_height(inputs.h); activities.set_height(inputs.h);
		for(uint i=0; i<inputs.h; i++) {
			const float* x = inputs.get_row(i);
			if(prototypes.h < K) {
				memcpy(prototypes.get_row(prototypes.h), inputs.get_row(i), inputs.w * sizeof(float));
				prototypes.set_height(prototypes.h+1);
			} else {
				int winner = compute_winner(x);
				winners(i,0) = (int)winner;
				winners(i,1) = compute_activity(dist(prototypes.get_row(winner), x, prototypes.w));
			}
		}

		// Compute output activities
		for(uint i=activities.h;i--;) {
			for(uint j=activities.w;j--;) {
				if(j>=prototypes.h) activities(i,j) = 0;
				else activities(i,j) = compute_activity(dist(prototypes.get_row(j),inputs.get_row(i),prototypes.w));
			}
		}
	}


private:
	float dist(const float* a, const float* b, uint dim) {
		float d = 0;
		for(uint i=dim; i--;) d += (a[i]-b[i])*(a[i]-b[i]);
		return d / dim;
	}

	int compute_winner(const float* x) {
		int winner = -1;
		float min_dist = FLT_MAX;
		for(uint k=0; k<prototypes.h; k++) {
			float d = dist(prototypes.get_row(k), x, prototypes.w);
			if(d <= min_dist) { min_dist = d; winner = k; }
		}
		adapt(winner, x);
		//DBG(min_dist);
		return winner;
	}

	inline float compute_activity(float dist) { return 1 - dist; }


	void adapt(int winner, const float* x) {
		float* mu = prototypes.get_row(winner);
		for(uint i=prototypes.w; i--;) mu[i] += eta * (x[i]-mu[i]);
	}
};



#endif /* SVQ_R_H_ */

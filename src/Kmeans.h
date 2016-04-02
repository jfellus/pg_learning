/*
 * Kmeans.h
 *
 *  Created on: 15 avr. 2015
 *      Author: jfellus
 */

#ifndef Kmeans_H_
#define Kmeans_H_

#include <pg.h>
#include <matrix.h>
#include <float.h>


/**
 * NOTE : Uses L_2 distance
 */
class Kmeans {
public:
	uint K;
	uint step;
	bool bFaster;

	Matrix prototypes;

	float MQE;
	uint emptyClusters;

	OUTPUT(float, MQE);
	OUTPUT(Matrix, prototypes);

private:
	Matrix tmp_prototypes;
	Matrix tmp_counts;

public:

	Kmeans() {
		K = 64;
		bFaster = true;
		step = 0;

		MQE = 0;
		emptyClusters = 0;
	}

	void init() {
		unlink("../MQE.txt");
	}

	void process(const Matrix& inputs) {
		if(!prototypes) {
			prototypes.init(K, inputs.w);
			tmp_prototypes.init(K, inputs.w);
			tmp_counts.init(K, 1);

			// Init prototypes to K first input vectors
			for(uint k=0; k<K; k++) memcpy(&prototypes(k,0), &inputs(k%inputs.h,0), prototypes.w*sizeof(float));
		}

		if(step==0) step = inputs.h / K;

		tmp_counts.clear();
		tmp_prototypes.clear();
		MQE = 0;

		if(bFaster) {
			size_t nbdone = 0;
			for(uint i=0; i<inputs.h; i+=step) {  process(&inputs(i,0)); nbdone++;}
			if(step > 1) step*=0.9;	if(step < 1) step = 1;
			MQE /= nbdone;
		}
		else {
			for(uint i = 0; i<inputs.h; i++) { process(&inputs(i,0));	}
			MQE /= inputs.h;
		}

		emptyClusters = 0;
		for(uint k=0; k<K; k++) {
			if(tmp_counts[k]==0) {memset(&prototypes(k,0), 0, prototypes.w*sizeof(float)); emptyClusters++;}
			else for(uint d=inputs.w;d--;) prototypes(k,d) = tmp_prototypes(k,d)/tmp_counts[k];
		}
		if(emptyClusters) DBG("Empty clusters : " << emptyClusters);

		dump_MQE();
	}


	void process(const float* v) {
		// Quantize input vector
		float mind = dist(&prototypes(0,0), v);
		uint winner = 0;
		for(uint k=1; k<prototypes.h; k++) {
			float d = dist(&prototypes(k,0), v);
			if(d<mind) { mind = d; winner = k; }
		}

		if(isnan(mind)) { DBG("Unquantizable point ! "); return; }

		// Update MQE
		MQE += mind;

		// Accumulate barycenter
		vec_add(&tmp_prototypes(winner,0), v);
		tmp_counts[winner]++;
	}

private:

	// Vector operations

	inline float dist(const float* a, const float* b) {
		float d = 0;
		for(uint i=prototypes.w; i--;) d += (a[i]-b[i])*(a[i]-b[i]);
		return d / prototypes.w;
	}

	inline void vec_add(float* a, const float* b) {
		for(uint i=prototypes.w;i--;) a[i] += b[i];
	}

	// DBG

	void dump_MQE() {
		FILE* f = fopen("../MQE.txt", "a");
		fprintf(f, "%f\n", MQE);
		fclose(f);
	}
};



#endif /* Kmeans_R_H_ */

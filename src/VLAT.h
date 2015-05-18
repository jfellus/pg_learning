/*
 * VLAT.h
 *
 *  Created on: 1 mai 2015
 *      Author: jfellus
 */

#ifndef VLAT_H_
#define VLAT_H_


#include <pg.h>
#include <matrix.h>
#include "Codebook.h"

class VLAT {
public:

	std::string codebook_path;

	Matrix signature;
	OUTPUT(Matrix, signature)

private:
	uint desc_dim;
	uint K;
	uint order;
	uint cluster_dim;

	Matrix winners;
	std::vector<Matrix> mean_tensors;

public:

	VLAT() {
		cluster_dim = 0;
		K = 0; order = 0; desc_dim = 0;
		codebook_path = "";
	}

	void init() {}

	void init(const Matrix& descriptors, const std::vector<Matrix>& mean_tensors) {
		order = mean_tensors.size();
		desc_dim = descriptors.w;
		K = mean_tensors[0].h;

		if(order==1) cluster_dim = desc_dim;
		else if(order==2) cluster_dim = (desc_dim*(desc_dim+1))/2;
		else ERROR("VLAT order > 2 not implemented yet !");

		signature.init(1,K * cluster_dim);
	}

	/** External quantizer */
	void process(const Matrix& descriptors, const Matrix& winners, const std::vector<Matrix>& mean_tensors) {
		if(!signature) {
			init(descriptors, mean_tensors);
			for(uint i=0; i<order; i++) this->mean_tensors.push_back(Matrix());
		}

		// Get clusters
		this->winners.map((Matrix&)winners);
		for(uint i=0; i<order; i++) this->mean_tensors[i].map((Matrix&)mean_tensors[i]);

		// Compute VLAT
		signature = 0;
		for(uint i=descriptors.h; i--;) append_vlat(descriptors.get_row(i), winners(i,0));
	}

	/** Load codebook from file (internal NN quantizer) */
	void process(const Matrix& descriptors) {
		if(!signature) {
			if(!codebook_path.empty()) ERROR("No codebook specified");
			read_matrix_list(codebook_path, mean_tensors);
			init(descriptors, mean_tensors);
		}

		compute_winners(descriptors);

		// Compute VLAT
		signature = 0;
		for(uint i=descriptors.h; i--;) append_vlat(descriptors.get_row(i), winners(i,0));
	}

	/** External codebook (internal NN quantizer) */
	void process(const Matrix& descriptors, const Codebook& codebook) {
		if(!signature) {
			init(descriptors, codebook.mean_tensors);
			for(uint i=0; i<order; i++) this->mean_tensors.push_back(Matrix());
			for(uint i=0; i<order; i++) this->mean_tensors[i].map((Matrix&)codebook.mean_tensors[i]);
		}

		compute_winners(descriptors);

		signature = 0;
		for(uint i=descriptors.h; i--;) append_vlat(descriptors.get_row(i), winners(i,0));
	}


private:

	void compute_winners(const Matrix& descriptors) {
		if(descriptors.h > winners.h) winners.free();
		if(!winners) winners.init(descriptors.h, 1);
		winners.set_height(descriptors.h);
		for(uint i=descriptors.h; i--;) winners(i,0) = compute_winner(descriptors.get_row(i));
	}

	void append_vlat(const float* descriptor, uint cluster) {
		if(cluster<0) return;
		float* sig = &signature[cluster*cluster_dim];
		if(order==1) vec_add(sig, descriptor, desc_dim);
		else if(order==2) vec_add_tensor2(sig, descriptor, desc_dim);
		vec_sub(sig, mean_tensors[order-1].get_row(cluster), cluster_dim);
	}



	// Internal NN quantizer

	float dist(const float* a, const float* b, uint dim) {
		float d = 0;
		for(uint i=dim; i--;) d += (a[i]-b[i])*(a[i]-b[i]);
		return d / dim;
	}

	int compute_winner(const float* x) {
		int winner = -1;
		float min_dist = FLT_MAX;
		for(uint k=0; k<mean_tensors[0].h; k++) {
			float d = dist(mean_tensors[0].get_row(k), x, mean_tensors[0].w);
			if(d <= min_dist) { min_dist = d; winner = k; }
		}
		return winner;
	}


	// Vector operations

	void vec_add(float* v1, const float* v2, uint dim) {for(uint i=dim; i--;) v1[i]+=v2[i];	}
	void vec_sub(float* v1, const float* v2, uint dim) {for(uint i=dim; i--;) v1[i]-=v2[i];	}

	void vec_add_tensor2(float* v1, const float* v2, uint dim) {
		uint k = 0;
		for(uint i=0; i<dim; i++) {
			for(uint j=i; j<dim; j++) {
				v1[k++] += v2[i]*v2[j];
			}
		}
	}

};



#endif /* VLAT_H_ */

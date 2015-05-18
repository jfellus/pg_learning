/*
 * Codebook.h
 *
 *  Created on: 1 mai 2015
 *      Author: jfellus
 */

#ifndef CODEBOOK_H_
#define CODEBOOK_H_


#include <pg.h>
#include <matrix.h>
#include <vector>

class Codebook {
public:
	std::string path;
	std::string out_file;
	bool learn;

	uint order;
	uint K;

	std::vector<Matrix> mean_tensors;
	OUTPUT(std::vector<Matrix>, mean_tensors);

private:
	size_t nb;

public:

	Codebook() { path=""; order=2; K=64; nb = 0; learn = true;}

	void init() {}

	/** Read codebook from file */
	void process() {
		if(mean_tensors.empty()) {
			if(path.empty()) ERROR("No codebook specified");
			read_matrix_list(path, mean_tensors);
			order = mean_tensors.size();
			K = mean_tensors[0].h;
		}
	}

	/** Compute codebook and save mean tensors to file
	 *  NOTE : If a <path> is provided, use its tensors instead of estimating */
	void process(const Matrix& descriptors, const Matrix& winners, bool bSave = false) {
		if(mean_tensors.empty()) {
			if(!path.empty() && !learn) {
				read_matrix_list(path, mean_tensors);
				order = mean_tensors.size();
				K = mean_tensors[0].h;
			} else {
				// Otherwise, start with fresh new tensors
				if(order>=1) {Matrix m(K, descriptors.w); m = 0; mean_tensors.push_back(m);}
				if(order>=2) {Matrix m(K, (descriptors.w*(descriptors.w+1))/2); m = 0; mean_tensors.push_back(m);}
			}
		}

		if(!path.empty() && !learn) return; // No learning if <path> provided

		if(order>=1) for(uint i=descriptors.h; i--;) vec_add(mean_tensors[0], descriptors.get_row(i), descriptors.w);
		if(order>=2) for(uint i=descriptors.h; i--;) vec_add_tensor2(mean_tensors[1], descriptors.get_row(i), descriptors.w);
		nb += descriptors.h;

		if(bSave && !out_file.empty()) {
			if(order>=1) vec_div(mean_tensors[0], nb, mean_tensors[0].w);
			if(order>=2) {
				vec_div(mean_tensors[1], nb, mean_tensors[1].w);
				vec_sub_tensor2(mean_tensors[1], mean_tensors[0], mean_tensors[0].w);
			}
			write_matrix_list(out_file, mean_tensors);
			DBG("Saved codebook to " << out_file);
			if(order>=2) {
				vec_add_tensor2(mean_tensors[1], mean_tensors[0], mean_tensors[0].w);
				vec_mul(mean_tensors[1], nb, mean_tensors[1].w);
			}
			if(order>=1) vec_mul(mean_tensors[0], nb, mean_tensors[0].w);
		}
	}


private:

	// Vector operations

	void vec_add(float* v1, const float* v2, uint dim) {for(uint i=dim; i--;) v1[i]+=v2[i];	}
	void vec_sub(float* v1, const float* v2, uint dim) {for(uint i=dim; i--;) v1[i]-=v2[i];	}
	void vec_div(float* v1, float f, uint dim) 		   {for(uint i=dim; i--;) v1[i]/=f; }
	void vec_mul(float* v1, float f, uint dim) 		   {for(uint i=dim; i--;) v1[i]*=f; }

	void vec_add_tensor2(float* v1, const float* v2, uint dim) {
		uint k = 0;
		for(uint i=0; i<dim; i++) {
			for(uint j=i; j<dim; j++) {
				v1[k++] += v2[i]*v2[j];
			}
		}
	}

	void vec_sub_tensor2(float* v1, const float* v2, uint dim) {
		uint k = 0;
		for(uint i=0; i<dim; i++) {
			for(uint j=i; j<dim; j++) {
				v1[k++] -= v2[i]*v2[j];
			}
		}
	}

};




#endif /* CODEBOOK_H_ */

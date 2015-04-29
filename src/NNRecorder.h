/*
 * NNRecorder.h
 *
 *  Created on: 28 avr. 2015
 *      Author: jfellus
 */

#ifndef NNRECORDER_H_
#define NNRECORDER_H_


#include <pg.h>
#include <matrix.h>
#include <float.h>
#include <vector>


class NNRecorder {
public:
	uint max;
	Matrix records;
	std::vector<ImageRGB> imgs;

	Matrix out;
	ImageRGB image;
	int winner;

	OUTPUT(Matrix, out);
	OUTPUT(ImageRGB, image);

public:
	NNRecorder() {
		max = 100;
		winner = -1;
	}

	void init() {
	}

	void process(const Matrix& in, bool bRecord = false) {
		if(!records) {
			records.init(max, in.n);
			out.init(1, in.n, &records[0]);
			records.set_height(0);
		}

		if(bRecord && records.h < max) {
			memcpy(records.get_row(records.h), in.data, in.n*sizeof(float));
			records.set_height(records.h+1);
		}

		winner = compute_winner(in.data);
		if(winner!=-1) {
			out.data = records.get_row(winner);
			out.set_height(1);
		} else {
			out.data = NULL;
			out.set_height(0);
		}
	}

	void process(const Matrix& in, const ImageRGB& img, bool bRecord = false) {
		process(in, bRecord);
		if(bRecord) {
			ImageRGB i; i.init(img.w, img.h);
			memcpy(i.data, img.data, img.w*img.h*3);
			imgs.push_back(i);
		}
		if(winner!=-1) {
			image.data = imgs[winner].data; image.w = imgs[winner].w; image.h = imgs[winner].h;
		} else {
			image.data = NULL; image.w = image.h = 0;
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
		for(uint k=0; k<records.h; k++) {
			float d = dist(records.get_row(k), x, records.w);
			if(d <= min_dist) { min_dist = d; winner = k; }
		}
		return winner;
	}
};


#endif /* NNRECORDER_H_ */

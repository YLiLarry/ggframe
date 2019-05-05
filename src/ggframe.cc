#include "ggframe.h"
#include <iostream>

#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace std;
using namespace ggframe;
using namespace cv;
using namespace cv::xfeatures2d;

Frame::Frame()
{
	m_cimg = make_unique<CImg<uint8_t>>();
	assert(m_cimg->width() == 0);
	assert(m_cimg->height() == 0);
}

Frame::Frame(unsigned w, unsigned h, unsigned d)
{
	m_cimg = make_unique<CImg<uint8_t>>(w, h, 1, d, 0);
}

Frame::Frame(filesystem::path filepath)
{
	m_cimg = make_unique<CImg<uint8_t>>(filepath.string().c_str());
}

void Frame::display() const
{
	m_cimg_display->resize(m_cimg->width(), m_cimg->height());
	m_cimg_display->display(*m_cimg);
}

void Frame::set(unsigned r, unsigned c, unsigned d, uint8_t v)
{
	m_cimg->operator()(c, r, 0, d) = v;
}

uint8_t Frame::get(unsigned r, unsigned c, unsigned d) const
{
	return m_cimg->operator()(c, r, 0, d);
}

void Frame::save(filesystem::path path)
{
	m_cimg->save(path.string().c_str());
}

void Frame::load(filesystem::path path)
{
	m_cimg->load(path.string().c_str());
}

unique_ptr<CImgDisplay> Frame::m_cimg_display = make_unique<CImgDisplay>();

InputEvent Frame::waitForInput()
{
	while (!m_cimg_display->is_closed()) {
		if (m_cimg_display->button()) { // Left button clicked
			m_cimg_display->set_button(m_cimg_display->button(), false);
			return InputEvent{ Mouse, m_cimg_display->button(), Press, mousePosition() };
		}
		if (m_cimg_display->key()) {
			m_cimg_display->set_key(m_cimg_display->key(), false);
			return InputEvent{ Keyboard, m_cimg_display->key(), Press, mousePosition() };
		}
		m_cimg_display->wait();
	}
	return InputEvent{ Window, 0, WindowClose, Pos{} };
}

Pos Frame::mousePosition()
{
	return Pos{ m_cimg_display->mouse_y(), m_cimg_display->mouse_x() };
}

void Frame::drawGrid()
{
	for (int r = 0; r < m_cimg->height(); r++) {
		for (int c = 0; c < m_cimg->width(); c++) {
			if (r % m_grid_size == 0 || c % m_grid_size == 0) {
				for (int d = 0; d < 3; d++) {
					set(r, c, d, min(get(r, c, d) + 25, 255));
				}
			}
		}
	}
}

unsigned Frame::gridSize()
{
	return m_grid_size;
}

void Frame::setGridSize(unsigned size)
{
	m_grid_size = size;
}

Rec::Rec(int top, int left, unsigned width, unsigned height)
	: m_t(top), m_l(left), m_w(width), m_h(height)
{

}

int Rec::left() const { return m_l; }
int Rec::right() const { return m_l + m_w - 1; }
int Rec::top() const { return m_t; }
int Rec::bottom() const { return m_t + m_h - 1; }
unsigned Rec::width() const { return m_w; }
unsigned Rec::height() const { return m_h; }

Rec Frame::bestGridRecCenteredAt(unsigned r, unsigned c, unsigned w, unsigned h)
{	
	/* calculate the number of cells needed */
	unsigned nc = w / m_grid_size + 1;
	unsigned nr = h / m_grid_size + 1;
	/* align size */
	w = nc * m_grid_size;
	h = nr * m_grid_size;
	/* find bounds */
	int left = c - w / 2;
	int top = r - h / 2;
	left = max(left, 0);
	top = max(top, 0);
	/* align bounds to grid */
	int gl = (left / m_grid_size) * m_grid_size;
	int gt = (top / m_grid_size) * m_grid_size;
	unsigned gr = gl + nc * m_grid_size - 1;
	unsigned gb = gt + nr * m_grid_size - 1;
	gr = min(gr, lastCol());
	gb = min(gb, lastRow());
	Rec unbounded = Rec(gt, gl, gr - gl + 1, gb - gt + 1);
	return unbounded.intersect(frameRec());
}

void Frame::drawRec(unsigned top, unsigned left, unsigned width, unsigned height)
{
	if (!width || !height) {
		return;
	}
	for (unsigned r = 0; r < height; r++) {
		set(top + r, left, 0, -1);
		set(top + r, left + width - 1, 0, -1);
	}
	for (unsigned c = 0; c < width; c++) {
		set(top, left + c, 0, -1);
		set(top + height - 1, left + c, 0, -1);;
	}
}

void Frame::drawRec(Rec const& rec)
{
	drawRec(rec.top(), rec.left(), rec.width(), rec.height());
}

unsigned Frame::lastCol() const
{
	return m_cimg->width() > 0 ? m_cimg->width() - 1 : 0;
}

unsigned Frame::lastRow() const
{
	return m_cimg->height() > 0 ? m_cimg->height() - 1 : 0;
}

unsigned Frame::nCols() const
{
	return m_cimg->width();
}

unsigned Frame::nRows() const
{
	return m_cimg->height();
}

void Frame::displaySift() const
{
	return displaySiftInRec(frameRec());
}

void Frame::showKeyPoints(vector<KeyPoint> const& keypoints) const
{
	Mat mat(nRows(), nCols(), CV_8U);
	for (int r = 0; r < nRows(); r++) {
		for (int c = 0; c < nCols(); c++) {
			uint8_t v = 0;
			for (int d = 0; d < m_cimg->depth(); d++) {
				v = max(v, get(r, c, d));
			}
			mat.at<uint8_t>(r, c) = v;
		}
	}
	Mat mat_keypts;
	drawKeypoints(mat, keypoints, mat_keypts);
	cv::imshow("img keypoints", mat_keypts);
	cv::waitKey(0);
}

void Frame::displaySiftInRec(Rec const& rec) const
{
	vector<KeyPoint> kps = getSiftKeyPointsInRec(rec);
	showKeyPoints(kps);
}

vector<KeyPoint> Frame::getSiftKeyPointsInRec(Rec const& rec) const
{
	shared_ptr<SIFT> sift = SIFT::create();
	Mat mat(nRows(), nCols(), CV_8U);
	Mat mask(nRows(), nCols(), CV_8U);
	for (int r = 0; r < nRows(); r++) {
		for (int c = 0; c < nCols(); c++) {
			uint8_t v = 0;
			for (int d = 0; d < m_cimg->depth(); d++) {
				v = max(v, get(r, c, v));
			}
			mat.at<uint8_t>(r, c) = v;
			if (rec.left() <= c && c <= rec.right()
				&& rec.top() <= r && r <= rec.bottom()) 
			{
				mask.at<uint8_t>(r, c) = 1;
			} else {
				mask.at<uint8_t>(r, c) = 0;
			}
		}
	}
	vector<KeyPoint> keypoints;
	sift->detect(mat, keypoints, mask);
	return keypoints;
}

bool Rec::empty() const
{
	return m_h == 0 || m_h == 0;
}

bool Frame::empty() const
{
	return m_cimg->height() == 0 || m_cimg->width() == 0;
}

cv::Mat Frame::cvMat() const
{
	Mat mat(nRows(), nCols(), CV_8U);
	for (int r = 0; r < nRows(); r++) {
		for (int c = 0; c < nCols(); c++) {
			uint8_t v = 0;
			for (int d = 0; d < m_cimg->depth(); d++) {
				v = max(v, get(r, c, d));
			}
			mat.at<uint8_t>(r, c) = v;
		}
	}
	return mat;
}

Rec Frame::findPattern(Frame const& pattern) const
{
	auto sift = SIFT::create();
	vector<KeyPoint> self_kps = getSiftKeyPointsInRec(frameRec());
	vector<KeyPoint> pattern_kps = pattern.getSiftKeyPointsInRec(pattern.frameRec());
	cv::Mat self_mat = cvMat();
	cv::Mat pattern_mat = pattern.cvMat();
	cv::Mat self_desc;
	cv::Mat pattern_desc;
	sift->compute(self_mat, self_kps, self_desc);
	sift->compute(pattern_mat, pattern_kps, pattern_desc);

	pattern.displaySift();
	displaySift();

	cv::BFMatcher bforce;
	vector<cv::DMatch> matches;
	bforce.add(self_desc);
	bforce.match(pattern_desc, matches);
	unsigned min_t = -1;
	unsigned max_b = 0;
	unsigned min_l = -1;
	unsigned max_r = 0;
	for (cv::DMatch& m : matches) {
		KeyPoint& pattern_kp = pattern_kps[m.queryIdx];
		KeyPoint& self_kp = self_kps[m.trainIdx];
		unsigned frame_col = self_kp.pt.x;
		unsigned frame_row = self_kp.pt.y;
		min_t = min(frame_row, min_t);
		max_b = max(frame_row, max_b);
		min_l = min(frame_col, min_l);
		max_r = max(frame_col, max_r);
	}
	Rec matched_rec(min_t, min_l, max_r - min_l + 1, max_b - min_t + 1);
	return matched_rec.intersect(frameRec());
}

Rec Frame::frameRec() const
{
	return Rec(0,0,nCols(),nRows());
}

Rec Rec::intersect(Rec const& other) const
{
	int l = max(left(), other.left());
	int t = max(top(), other.top());
	int r = min(right(), other.right());
	int b = min(bottom(), other.bottom());
	return Rec(t, l, r - l + 1, b - t + 1);
}

Frame::Frame(Frame const& other)
{
	m_grid_size = other.m_grid_size;
	m_cimg = make_unique<CImg<uint8_t>>(*other.m_cimg);
}

Frame Frame::cutRec(Rec const& rec) const
{
	Frame output(rec.width(), rec.height(), m_cimg->depth());
	for (int row = 0; row < rec.height(); row++) {
		for (int col = 0; col < rec.width(); col++) {
			for (int depth = 0; depth < m_cimg->depth(); depth++) {
				uint8_t color = get(rec.top() + row, rec.left() + col, depth);
				output.set(row, col, depth, color);
			}
		}
	}
	return output;
}

Frame& Frame::operator=(Frame const& other)
{
	m_grid_size = other.m_grid_size;
	m_cimg = make_unique<CImg<uint8_t>>(*other.m_cimg);
	return *this;
}

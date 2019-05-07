#include <ggframe.h>
#include <iostream>

#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace std;
using namespace ggframe;
using namespace cv;
using namespace cv::xfeatures2d;

#if APPLE
	using boost::filesystem::path;
	using std::shared_ptr;
#else
	using std::filesystem::path;
#endif

Frame::Frame()
{
	m_image = make_unique<image_t>();
	assert(nCols() == 0);
	assert(nRows() == 0);
}

Frame::Frame(unsigned nrows, unsigned ncols)
{
	m_image = make_unique<image_t>(nrows, ncols, CV_8UC4, 0);
}

Frame::Frame(path filepath)
{
	m_image = make_unique<image_t>();
	load(filepath);
}

void Frame::display() const
{
	static string window_title = "";
	if (window_title.length() == 0) {
		window_title = "ggcapture";
		cv::namedWindow(window_title);
	}
	cv::imshow("ggcapture", *m_image);
	cv::waitKey(1);
}

unsigned Frame::colorIndex(Color color) const
{
	if (color == Color::A) {
		return 3;
	}
	if (color == Color::R) {
		return 2;
	}
	if (color == Color::G) {
		return 1;
	}
	if (color == Color::B) {
		return 0;
	}
	return 0;
}

void Frame::set(unsigned r, unsigned c, Color color, uint8_t v)
{
	cv::Vec4b& vec = m_image->at<cv::Vec4b>(r,c);
	vec[colorIndex(color)] = v;
}

uint8_t Frame::get(unsigned r, unsigned c, Color color) const
{
	return m_image->at<cv::Vec4b>(r,c)[colorIndex(color)];
}

void Frame::save(path path)
{
	cv::imwrite(path.string().c_str(), *m_image);
}

void Frame::load(path path)
{
	*m_image = cv::imread(path.string().c_str());
}

InputEvent Frame::waitForInput()
{
	// while (!m_image_display->is_closed()) {
	// 	if (m_image_display->button()) { // Left button clicked
	// 		m_image_display->set_button(m_image_display->button(), false);
	// 		return InputEvent{ Mouse, m_image_display->button(), Press, mousePosition() };
	// 	}
	// 	if (m_image_display->key()) {
	// 		m_image_display->set_key(m_image_display->key(), false);
	// 		return InputEvent{ Keyboard, m_image_display->key(), Press, mousePosition() };
	// 	}
	// 	m_image_display->wait();
	// }
	return InputEvent{ Window, 0, WindowClose, Pos{} };
}

Pos Frame::mousePosition()
{
	return Pos{ 0,0 };
}

void Frame::drawGrid()
{
	for (int r = 0; r < nRows(); r++) {
		for (int c = 0; c < nCols(); c++) {
			if (r % m_grid_size == 0 || c % m_grid_size == 0) {
				for (int d = 0; d < nColors(); d++) {
					set(r, c, static_cast<Color>(d), min(get(r, c, static_cast<Color>(d)) + 25, 255));
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
		set(top + r, left, Color::R, -1);
		set(top + r, left + width - 1, Color::R, -1);
	}
	for (unsigned c = 0; c < width; c++) {
		set(top, left + c, Color::R, -1);
		set(top + height - 1, left + c, Color::R, -1);;
	}
}

void Frame::drawRec(Rec const& rec)
{
	drawRec(rec.top(), rec.left(), rec.width(), rec.height());
}

unsigned Frame::lastCol() const
{
	return nCols() > 0 ? nCols() - 1 : 0;
}

unsigned Frame::lastRow() const
{
	return nRows() > 0 ? nRows() - 1 : 0;
}

unsigned Frame::nCols() const
{
	return m_image->size().width;
}

unsigned Frame::nRows() const
{
	return m_image->size().height;
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
			for (int d = 0; d < nColors(); d++) {
				v = max(v, get(r, c, static_cast<Color>(d)));
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
			for (int d = 0; d < nColors(); d++) {
				v = max(v, get(r, c, static_cast<Color>(d)));
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
	return nRows() == 0 || nCols() == 0;
}

cv::Mat Frame::cvMat() const
{
	Mat mat(nRows(), nCols(), CV_8U);
	for (int r = 0; r < nRows(); r++) {
		for (int c = 0; c < nCols(); c++) {
			uint8_t v = 0;
			for (int d = 0; d < nColors(); d++) {
				v = max(v, get(r, c, static_cast<Color>(d)));
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
	m_image = make_unique<image_t>(*other.m_image);
}

Frame Frame::cutRec(Rec const& rec) const
{
	Frame output(rec.width(), rec.height());
	for (int row = 0; row < rec.height(); row++) {
		for (int col = 0; col < rec.width(); col++) {
			for (int depth = 0; depth < nColors(); depth++) {
				uint8_t color = get(rec.top() + row, rec.left() + col, static_cast<Color>(depth));
				output.set(row, col, static_cast<Color>(depth), color);
			}
		}
	}
	return output;
}

void Frame::crop(Rec const& rec)
{
	cv::Range row_range(rec.top(), rec.bottom());
	cv::Range col_range(rec.left(), rec.right());
	m_image = make_unique<image_t>(*m_image, row_range, col_range);
}

Frame& Frame::operator=(Frame const& other)
{
	m_grid_size = other.m_grid_size;
	m_image = make_unique<image_t>(*other.m_image);
	return *this;
}

ostream& ggframe::operator<<(ostream& out, Rec const& rec)
{
	out << "Rec { left:" << rec.left() << " right:" << rec.right() 
		<< " top:" << rec.top() << " bottom:" << rec.bottom()
		<< " width:" << rec.width() << " height:" << rec.height() 
		<< " }";
	return out;
}

void Frame::resize(unsigned width, unsigned height)
{
	m_image->resize(width, height);
}

unsigned Frame::nColors() const
{
	return 3;
}

uint8_t* Frame::data() const
{
	return m_image->data;
}

uint8_t* Frame::data()
{
	return m_image->data;
}


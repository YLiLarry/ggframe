#include <ggframe.h>
#include <iostream>

#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace ggframe;
using namespace cv::xfeatures2d;
using ggframe::Size;

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
		window_title = "ggframe";
		cv::namedWindow(window_title);
	}
	cv::imshow("ggframe", *m_image);
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
	struct CV_SetMouseCallBack_UserData_Wrapper {
		InputEvent input_event;
		bool waiting_for_input;
	};
	CV_SetMouseCallBack_UserData_Wrapper userdata_wrapper;
	userdata_wrapper.waiting_for_input = true;
	cv::setMouseCallback("ggframe", [](int event, int x, int y, int flags, void* userdata) {
		auto userdata_wrapper = static_cast<CV_SetMouseCallBack_UserData_Wrapper*>(userdata);
		if (event == cv::EVENT_LBUTTONDOWN) {
			userdata_wrapper->input_event = InputEvent{ Mouse, MouseLeft, Press, Pos::rc(y,x) };
			userdata_wrapper->waiting_for_input = false;
		}
	}, &userdata_wrapper);
	while (userdata_wrapper.waiting_for_input) {
		cv::waitKey(1);
	}
	return userdata_wrapper.input_event;
}

void Frame::drawGrid()
{
	for (int r = 0; r < nRows(); r++) {
		for (int c = 0; c < nCols(); c++) {
			if (r % m_grid_size == 0 || c % m_grid_size == 0) {
				set(r, c, Color::R, min(get(r, c, Color::R) + 25, 255));
				set(r, c, Color::B, min(get(r, c, Color::B) + 25, 255));
				set(r, c, Color::G, min(get(r, c, Color::G) + 25, 255));
			}
		}
	}
}

unsigned Frame::gridSize() const { return m_grid_size; }

void Frame::setGridSize(unsigned size)
{
	m_grid_size = size;
}

int Rec::left() const { return m_l; }
int Rec::right() const { return m_r; }
int Rec::top() const { return m_t; }
int Rec::bottom() const { return m_b; }
unsigned Rec::width() const { return m_w; }
unsigned Rec::height() const { return m_h; }

int Pos::row() const { return m_r; }
int Pos::col() const { return m_c; }

unsigned ggframe::Size::height() const { return m_h; }
unsigned ggframe::Size::width() const { return m_w; }

Rec Frame::bestGridRecCenteredAt(Pos const& pt, ggframe::Size const& size)
{	
	/* calculate the number of cells needed */
	unsigned cells_ncols = size.width() / gridSize();
    unsigned cells_nrows = size.height() / gridSize();
    if (size.width() % gridSize() > 0) {
        cells_ncols++;
    }
    if (size.height() % gridSize() > 0) {
        cells_nrows++;
    }
    unsigned clicked_cell_row = pt.row() / gridSize();
    unsigned clicked_cell_col = pt.col() / gridSize();
    unsigned left_cell = clicked_cell_col - cells_ncols / 2;
    unsigned top_cell = clicked_cell_row - cells_nrows / 2;
    unsigned right_cell = left_cell + cells_ncols - 1;
    unsigned bottom_cell = top_cell + cells_nrows - 1;
    int left = left_cell * gridSize();
    int top = top_cell * gridSize();
    int bottom = (bottom_cell + 1) * gridSize();
    int right = (right_cell + 1) * gridSize();
	Rec unbounded = Rec::tlbr(top, left, bottom, right);
	return unbounded.intersect(frameRec());
}

Pos Pos::rc(int r, int c)
{
    Pos rtv;
    rtv.m_r = r;
    rtv.m_c = c;
    return rtv;
}

ggframe::Size ggframe::Size::hw(int h, int w)
{
    Size rtv;
    rtv.m_h = h;
    rtv.m_w = w;
    return rtv;
}

void Frame::drawRec(Rec const& rec)
{
    cv::rectangle(*m_image, cv::Point(rec.left(), rec.top()), cv::Point(rec.right(), rec.bottom()), cv::Scalar(0,0,255));
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
	return m_image->cols;
}

unsigned Frame::nRows() const
{
	return m_image->rows;
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
			v = max(v, get(r, c, Color::R));
			v = max(v, get(r, c, Color::B));
			v = max(v, get(r, c, Color::G));
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
			v = max(v, get(r, c, Color::R));
			v = max(v, get(r, c, Color::B));
			v = max(v, get(r, c, Color::G));
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
			v = max(v, get(r, c, Color::R));
			v = max(v, get(r, c, Color::B));
			v = max(v, get(r, c, Color::G));
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
	Rec matched_rec = Rec::tlbr(min_t, min_l, max_b, max_r);
	return matched_rec.intersect(frameRec());
}

Rec Rec::tlbr(int top, int left, int bottom, int right)
{
    Rec rtv;
    rtv.m_t = top;
    rtv.m_l = left;
    rtv.m_b = bottom;
    rtv.m_r = right;
    rtv.m_h = bottom - top + 1;
    rtv.m_w = right - left + 1;
    return rtv;
}

Rec Frame::frameRec() const
{
	return Rec::tlbr(0,0,lastRow(),lastCol());
}

Rec Rec::intersect(Rec const& other) const
{
	int l = max(left(), other.left());
	int t = max(top(), other.top());
	int r = min(right(), other.right());
	int b = min(bottom(), other.bottom());
	return Rec::tlbr(t, l, b, r);
}

Frame::Frame(Frame const& other)
{
	m_grid_size = other.m_grid_size;
	m_image = make_unique<image_t>(*other.m_image);
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

void Frame::resize(ggframe::Size const& size)
{
	m_image->resize(size.width(), size.height());
}

uint8_t* Frame::data() const
{
	return m_image->data;
}

uint8_t* Frame::data()
{
	return m_image->data;
}


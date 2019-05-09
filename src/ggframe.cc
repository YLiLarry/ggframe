#include <ggframe.h>
#include <iostream>

#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace ggframe;
using namespace cv::xfeatures2d;
using ggframe::Size;

#if APPLE
	using boost;
	using std::shared_ptr;
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

Frame::Frame(filesystem::path filepath)
{
	m_image = make_unique<image_t>();
	load(filepath);
}

void Frame::display(string const& title) const
{
	string window_title = title.size() ? title : "ggframe";
	cv::imshow(window_title, *m_image);
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

void Frame::save(filesystem::path path)
{
	cv::imwrite(path.string().c_str(), *m_image);
}

void Frame::load(filesystem::path path)
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
		int key = cv::waitKey(1);
		if (key >= 0) {
			cv:setMouseCallback("ggframe", nullptr);
			userdata_wrapper.waiting_for_input = false;
			userdata_wrapper.input_event = InputEvent{ Keyboard, static_cast<unsigned>(key), Press, Pos() };
			break;
		}
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

ggframe::Size ggframe::Size::hw(unsigned h, unsigned w)
{
    assert(h > 0);
    assert(w > 0);
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

bool Rec::empty() const
{
	return m_h == 0 || m_h == 0;
}

bool Frame::empty() const
{
	return nRows() == 0 || nCols() == 0;
}

cv::Mat& Frame::cvMat() { return *m_image; }
cv::Mat const& Frame::cvMat() const { return *m_image; }

Rec Frame::recMatchedTemplate(Frame const& pattern) const
{
	/* open fucking cv api is as follows in its code, 
		the query image is also known as image1, meaning the template image (usually smaller)
		the train image is also known as image2, meaning the scene image that contains the templated object 
	*/
	auto sift = SIFT::create(0,3,0.08,10,0.8);

	/* compute keypoints and descriptors */
	cv::Mat query_desc;
	cv::Mat train_desc;
	vector<KeyPoint> query_keypoints;
	vector<KeyPoint> train_keypoints;
	sift->detectAndCompute(*pattern.m_image, cv::noArray(), query_keypoints, query_desc);
	sift->detectAndCompute(*m_image, cv::noArray(), train_keypoints, train_desc);

	/* filter matches */
	vector<cv::DMatch> good_matches;
	vector<vector<char>> match_mask;
	
	auto bforce = cv::BFMatcher::create(NORM_L2, false);
	vector<vector<cv::DMatch>> matches;
	bforce->add(train_desc);
	bforce->knnMatch(query_desc, train_desc, matches, 2);

	for (vector<cv::DMatch> const& m : matches) {
		cv::DMatch const& fst_match = m[0];
		cv::DMatch const& snd_match = m[1];
		vector<char> local_mask;
		if (fst_match.distance / snd_match.distance < 0.8) {
			good_matches.push_back(fst_match);
			local_mask.push_back(1);
		} else {
			local_mask.push_back(0);
		}
		local_mask.push_back(0);
		match_mask.push_back(local_mask);
	}

	/* draw matches */
	vector<cv::KeyPoint> good_query_keypoints;
	vector<cv::KeyPoint> good_train_keypoints;
	for (cv::DMatch const& m : good_matches) {
		good_query_keypoints.push_back(query_keypoints[m.queryIdx]);
		good_train_keypoints.push_back(train_keypoints[m.trainIdx]);
	}

	assert(good_query_keypoints.size() > 0);

	Frame canvas;
	cv::drawMatches(*pattern.m_image, query_keypoints, *m_image, train_keypoints, matches, *canvas.m_image, cv::Scalar::all(-1), cv::Scalar::all(-1), match_mask);
	canvas.display();
	std::cerr << "showing filtered matches" << std::endl;
	canvas.waitForInput();

	Rec matched_rec;
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
	m_image = make_unique<image_t>(other.m_image->clone());
}

void Frame::crop(Rec const& rec)
{
	cv::Range row_range(rec.top(), rec.bottom());
	cv::Range col_range(rec.left(), rec.right());
	cv::Mat ref_mat = cv::Mat(*m_image, row_range, col_range);
	m_image = make_unique<image_t>(ref_mat.clone());
}

Frame& Frame::operator=(Frame const& other)
{
	m_grid_size = other.m_grid_size;
	m_image = make_unique<image_t>(other.m_image->clone());
    m_keypoints = other.m_keypoints;
    m_descriptors = other.m_descriptors.clone();
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

void Frame::computeKeypoints(cv::Ptr<cv::Feature2D> extractor)
{
    extractor->detectAndCompute(*m_image, cv::noArray(), m_keypoints, m_descriptors);
}

vector<cv::KeyPoint>& Frame::keypoints() { return m_keypoints; }
vector<cv::KeyPoint> const& Frame::keypoints() const { return m_keypoints; }

cv::Mat& Frame::descriptors() { return m_descriptors; }
cv::Mat const& Frame::descriptors() const { return m_descriptors; }

bool Rec::containsPos(Pos const& pos) const
{
    return left() <= pos.col() && pos.col() <= right()
           && top() <= pos.row() && pos.row() <= bottom();
}

bool Frame::hasKeypoints() const
{
    return (! m_keypoints.empty()) && (! m_descriptors.empty());
}

void Frame::makeContinuous()
{
    if (! m_image->isContinuous()) {
        *m_image = m_image->clone();
    }
    assert(m_image->isContinuous());
}

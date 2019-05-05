#include <CImg.h>
#include <memory>
#include <filesystem>

#include <opencv2/core.hpp>

namespace ggframe
{
	using namespace cimg_library;
	using namespace std;
	using namespace cv;

	struct Pos
	{
		int r;
		int c;
	};

	class Rec
	{
		int m_l = 0;
		int m_t = 0;
		unsigned m_w = 0;
		unsigned m_h = 0;
	public:
		unsigned width() const;
		unsigned height() const;
		int left() const;
		int right() const;
		int top() const;
		int bottom() const;
		Rec() = default;
		Rec(int top, int left, unsigned width, unsigned height);
		Rec intersect(Rec const& other) const;
		bool empty() const;
	};

	enum InputButton {
		MouseLeft, MouseMid, MouseRight,
		KeyEnter,
	};

	enum InputSource {
		Mouse, Keyboard, Window
	};

	enum InputEventType {
		Press, Release, WindowClose
	};

	struct InputEvent
	{
		InputSource source;
		unsigned keyCode;
		InputEventType type;
		Pos mouse;
	};

	class Frame
	{
	private:
		unique_ptr<CImg<uint8_t>> m_cimg;
		static unique_ptr<CImgDisplay> m_cimg_display;
		int m_grid_size = 1;
		vector<KeyPoint> getSiftKeyPointsInRec(Rec const& rec) const;
		void showKeyPoints(vector<cv::KeyPoint> const& keypoints) const;
		cv::Mat cvMat() const;

	public:
		Frame();
		Frame(unsigned w, unsigned h, unsigned d);
		Frame(filesystem::path filepath);
		Frame(Frame const& other);
		Frame& operator=(Frame const& other);
		void set(unsigned r, unsigned c, unsigned d, uint8_t v);
		uint8_t get(unsigned r, unsigned c, unsigned d) const;
		unsigned lastCol() const;
		unsigned lastRow() const;
		unsigned nCols() const;
		unsigned nRows() const;
		void display() const;
		void drawGrid();
		void setGridSize(unsigned size);
		unsigned gridSize();
		void drawRec(unsigned r, unsigned c, unsigned w, unsigned h);
		void drawRec(Rec const& rec);
		void displaySift() const;
		void displaySiftInRec(Rec const& rec) const;
		InputEvent waitForInput();
		Pos mousePosition();
		void save(filesystem::path path);
		void load(filesystem::path path);
		Rec bestGridRecCenteredAt(unsigned r, unsigned c, unsigned w, unsigned h);
		Rec frameRec() const;
		Rec findPattern(Frame const& pattern) const;
		Frame cutRec(Rec const& rec) const;
		bool empty() const;
	};
}

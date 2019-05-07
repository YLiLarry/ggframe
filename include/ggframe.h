#include <memory>
#include <opencv2/core.hpp>

#if APPLE
#include <boost/filesystem.hpp>
#else
#include <filesystem>
#endif

namespace ggframe
{
	using namespace std;
	using namespace cv;

#if APPLE
	using boost::filesystem::path;
	using std::shared_ptr;
#else
	using std::filesystem::path;
#endif

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
		friend ostream& operator<<(ostream& out, Rec const& rec);
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

	enum Color {
		R, G, B, A
	};
	 
	class Frame
	{
		typedef cv::Mat image_t;
	private:
		unique_ptr<image_t> m_image;
		int m_grid_size = 1;
		vector<KeyPoint> getSiftKeyPointsInRec(Rec const& rec) const;
		void showKeyPoints(vector<cv::KeyPoint> const& keypoints) const;
		cv::Mat cvMat() const;
		unsigned colorIndex(Color color) const;
		unsigned nColors() const;

	public:
		Frame();
		Frame(unsigned nrows, unsigned ncols);
		Frame(path filepath);
		Frame(Frame const& other);
		Frame& operator=(Frame const& other);
		void set(unsigned r, unsigned c, Color color, uint8_t v);
		void setBGR(unsigned r, unsigned c, uint8_t v);
		uint8_t get(unsigned r, unsigned c, Color color) const;
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
		void save(path filepath);
		void load(path filepath);
		Rec bestGridRecCenteredAt(unsigned r, unsigned c, unsigned w, unsigned h);
		Rec frameRec() const;
		Rec findPattern(Frame const& pattern) const;
		Frame cutRec(Rec const& rec) const;
		void crop(Rec const& rec);
		bool empty() const;
		void resize(unsigned width, unsigned height);
		uint8_t* data() const;
		uint8_t* data();
	};
}

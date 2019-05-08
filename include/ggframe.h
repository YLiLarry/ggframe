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

    class Pos
    {
        int m_r;
        int m_c;
    public:
        Pos() = default;
        static Pos rc(int r, int c);
        int row() const;
        int col() const;
    };

    class Size
    {
        int m_h;
        int m_w;
    public:
        Size() = default;
        static Size hw(int h, int w);
        unsigned height() const;
        unsigned width() const;
    };

    class Rec
    {
        int m_l = 0;
        int m_t = 0;
        int m_b = 0;
        int m_r = 0;
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
        static Rec tlbr(int top, int left, int bottom, int right);
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

    public:
        Frame();
        Frame(unsigned nrows, unsigned ncols);
        Frame(path filepath);
        Frame(Frame const& other);
        Frame& operator=(Frame const& other);
        void set(unsigned r, unsigned c, Color color, uint8_t v);
        uint8_t get(unsigned r, unsigned c, Color color) const;
        unsigned lastCol() const;
        unsigned lastRow() const;
        unsigned nCols() const;
        unsigned nRows() const;
        void display() const;
        void drawGrid();
        void setGridSize(unsigned size);
        unsigned gridSize() const;
        void drawRec(Rec const& rec);
        void displaySift() const;
        void displaySiftInRec(Rec const& rec) const;
        InputEvent waitForInput();
        void save(path filepath);
        void load(path filepath);
        Rec bestGridRecCenteredAt(Pos const&, Size const&);
        Rec frameRec() const;
        Rec findPattern(Frame const& pattern) const;
        void crop(Rec const& rec);
        bool empty() const;
        void resize(Size const& size);
        uint8_t* data() const;
        uint8_t* data();
    };
}

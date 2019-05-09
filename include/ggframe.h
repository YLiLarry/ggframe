#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/features2d/features2d.hpp>

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
    using namespace boost;
    using std::shared_ptr;
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
        unsigned m_h;
        unsigned m_w;
    public:
        Size() = default;
        static Size hw(unsigned h, unsigned w);
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
        bool containsPos(Pos const& pos) const;
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

    enum InputAction {
        Press, Release, WindowClose
    };

    struct InputEvent
    {
        InputSource source;
        unsigned keyCode;
        InputAction type;
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
        vector<cv::KeyPoint> m_keypoints;
        cv::Mat m_descriptors;

    protected:
        unsigned colorIndex(Color color) const;

    public:
        Frame();
        Frame(unsigned nrows, unsigned ncols);
        Frame(filesystem::path filepath);
        Frame(Frame const& other);
        Frame& operator=(Frame const& other);
        void set(unsigned r, unsigned c, Color color, uint8_t v);
        uint8_t get(unsigned r, unsigned c, Color color) const;
        unsigned lastCol() const;
        unsigned lastRow() const;
        unsigned nCols() const;
        unsigned nRows() const;
        void display(string const& title = "") const;
        void drawGrid();
        void setGridSize(unsigned size);
        unsigned gridSize() const;
        void drawRec(Rec const& rec);
        InputEvent waitForInput();
        void save(filesystem::path filepath);
        void load(filesystem::path filepath);
        Rec bestGridRecCenteredAt(Pos const&, Size const&);
        Rec frameRec() const;
        Rec recMatchedTemplate(Frame const& pattern) const; /* deprecated */
        void crop(Rec const& rec);
        bool empty() const;
        void resize(Size const& size);

        cv::Mat& cvMat();
        cv::Mat const& cvMat() const;
        uint8_t* data() const;
        uint8_t* data();
        void computeKeypoints(cv::Ptr<cv::Feature2D> extractor);
        vector<cv::KeyPoint>& keypoints();
        vector<cv::KeyPoint> const& keypoints() const;
        cv::Mat& descriptors();
        cv::Mat const& descriptors() const;
        bool hasKeypoints() const;
        void makeContinuous();
    };
}

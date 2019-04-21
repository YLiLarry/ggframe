#include <CImg.h>
#include <memory>
#include <filesystem>

namespace ggframe
{
	using namespace cimg_library;
	using namespace std;

	class Frame
	{
	private:
		unique_ptr<CImg<uint8_t>> m_cimg;
		unique_ptr<CImgDisplay> m_cimg_display;
	public:
		Frame(size_t w, size_t h, size_t d);
		Frame(Frame const&) = delete;
		Frame& operator=(Frame const&) = delete;
		void set(size_t x, size_t y, size_t d, uint8_t v);
		void display();
		void save(filesystem::path path);
	};
}

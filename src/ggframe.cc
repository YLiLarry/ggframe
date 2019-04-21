#include "ggframe.h"

using namespace std;
using namespace ggframe;

Frame::Frame(size_t w, size_t h, size_t d)
{
	m_cimg = make_unique<CImg<uint8_t>>(w,h,1,d,0);
}

void Frame::display()
{
	if (m_cimg_display == nullptr)
	{
		m_cimg_display = make_unique<CImgDisplay>();
	}
	m_cimg_display->resize(m_cimg->width(), m_cimg->height());
	m_cimg_display->display(*m_cimg);
}

void Frame::set(size_t x, size_t y, size_t d, uint8_t v)
{
	m_cimg->operator()(x, y, 0, d) = v;
}

void Frame::save(filesystem::path path)
{
	m_cimg->save(path.string().c_str());
}

void Frame::load(filesystem::path path)
{
	m_cimg->load(path.string().c_str());
}

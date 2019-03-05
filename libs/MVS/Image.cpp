/*
* Image.cpp
*/

#include "Common.h"
#include "Image.h"

using namespace MVS;


// D E F I N E S ///////////////////////////////////////////////////


// S T R U C T S ///////////////////////////////////////////////////

// 打开图像，返回图像的首指针
IMAGEPTR Image::OpenImage(const String& fileName)
{
	#if 0
	if (Util::isFullPath(fileName))
		return IMAGEPTR(CImage::Create(fileName, CImage::READ));
	return IMAGEPTR(CImage::Create((Util::getCurrentFolder()+fileName).c_str(), CImage::READ));
	#else
	return IMAGEPTR(CImage::Create(fileName, CImage::READ));
	#endif
} // OpenImage
/*----------------------------------------------------------------*/

// 返回图像头指针
IMAGEPTR Image::ReadImageHeader(const String& fileName)
{
	IMAGEPTR pImage(OpenImage(fileName));
	if (pImage == NULL || FAILED(pImage->ReadHeader())) {   // pImage->ReadHeader() { return ok; }
		LOG("（libs/MVS/Image.cpp）error: failed loading image header");
		pImage.Release();
	}
	return pImage;
} // ReadImageHeader
/*----------------------------------------------------------------*/

// 返回图像头指针
IMAGEPTR Image::ReadImage(const String& fileName, Image8U3& image)
{
	IMAGEPTR pImage(OpenImage(fileName));
	if (pImage != NULL && !ReadImage(pImage, image))
		pImage.Release();
	return pImage;
} // ReadImage
/*----------------------------------------------------------------*/

bool Image::ReadImage(IMAGEPTR pImage, Image8U3& image)
{
	if (FAILED(pImage->ReadHeader())) {
		LOG("（libs/MVS/Image.cpp）error: failed loading image header");
		return false;
	}
	image.create(pImage->GetHeight(), pImage->GetWidth());
	if (FAILED(pImage->ReadData(image.data, PF_R8G8B8, 3, (CImage::Size)image.step))) {
		LOG("（libs/MVS/Image.cpp）error: failed loading image data");
		return false;
	}
	return true;
} // ReadImage
/*----------------------------------------------------------------*/


bool Image::LoadImage(const String& fileName, unsigned nMaxResolution)
{
	name = fileName;
	// open image file
	IMAGEPTR pImage(OpenImage(fileName));
	if (pImage == NULL) {
		LOG("（libs/MVS/Image.cpp）error: failed opening input image '%s'", name.c_str());
		return false;
	}
	// create and fill image data
	if (!ReadImage(pImage, image)) {
		LOG("（libs/MVS/Image.cpp）error: failed loading image '%s'", name.c_str());
		return false;
	}
	// resize image if needed
	scale = ResizeImage(nMaxResolution);
	return true;
} // LoadImage
/*----------------------------------------------------------------*/

// open the stored image file name and read again the image data 打开存储的图像文件名并再次读取图像数据
bool Image::ReloadImage(unsigned nMaxResolution, bool bLoadPixels)
{
	IMAGEPTR pImage(bLoadPixels ? ReadImage(name, image) : ReadImageHeader(name));  // 根据是否加载像素判断是读取图像还是读取图像头文件
	if (pImage == NULL) {
		LOG("（libs/MVS/Image.cpp）error: failed reloading image '%s'", name.c_str());
		return false;
	}
	if (!bLoadPixels) { // 如果没有加载像素，则初始化图像大小
		// init image size
		width = pImage->GetWidth();
		height = pImage->GetHeight();
	}
	// resize image if needed
	scale = ResizeImage(nMaxResolution);
	return true;
} // ReloadImage
/*----------------------------------------------------------------*/

// free the image data
void Image::ReleaseImage()
{
	image.release();
} // ReleaseImage
/*----------------------------------------------------------------*/

// resize image if needed
// return scale
float Image::ResizeImage(unsigned nMaxResolution)
{
	if (!image.empty()) {
		width = image.width();
		height = image.height();
	}
	if (nMaxResolution == 0 || MAXF(width,height) <= nMaxResolution)
		return 1.f;
	float scale;
	if (width > height) {
		scale = (float)nMaxResolution/width;
		height = height*nMaxResolution/width;
		width = nMaxResolution;
	} else {
		scale = (float)nMaxResolution/height;
		width = width*nMaxResolution/height;
		height = nMaxResolution;
	}
	if (!image.empty())
		cv::resize(image, image, cv::Size((int)width, (int)height), 0, 0, cv::INTER_AREA);
	return scale;
} // ResizeImage
/*----------------------------------------------------------------*/

// compute image scale for a given max and min resolution, using the current image file data
// 使用当前图像文件数据计算给定最大和最小分辨率的图像比例
unsigned Image::RecomputeMaxResolution(unsigned& level, unsigned minImageSize) const
{
	IMAGEPTR pImage(ReadImageHeader(name));
	if (pImage == NULL) {
		// something went wrong, use the current known size (however it will most probably fail later)
		return Image8U3::computeMaxResolution(width, height, level, minImageSize);
	}
	// re-compute max image size
	return Image8U3::computeMaxResolution(pImage->GetWidth(), pImage->GetHeight(), level, minImageSize);
} // RecomputeMaxResolution
/*----------------------------------------------------------------*/


// 从平台姿态和相对于平台的相机姿态计算相机外物
// 从归一化变为非归一化并计算投影矩阵
Camera Image::GetCamera(const PlatformArr& platforms, const Image8U::Size& resolution) const
{
	ASSERT(platformID != NO_ID);
	ASSERT(cameraID != NO_ID);
	ASSERT(poseID != NO_ID);

	// 计算归一化绝对相机姿态
	const Platform& platform = platforms[platformID];
	Camera camera(platform.GetCamera(cameraID, poseID));	// 返回归一化绝对相机姿态

	// 计算非归一化的相机姿态
	camera.K = camera.GetK<REAL>(resolution.width, resolution.height);
	camera.ComposeP();	// 计算投影矩阵

	return camera;
} // GetCamera
void Image::UpdateCamera(const PlatformArr& platforms)
{
	camera = GetCamera(platforms, Image8U::Size(width, height));
} // UpdateCamera
/*----------------------------------------------------------------*/
